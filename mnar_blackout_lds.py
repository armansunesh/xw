# ============================================================
# Module: mnar_blackout_lds
# ------------------------------------------------------------
# - MNAR-aware linear dynamical system for traffic blackouts
# - Extended Kalman filter using speeds x_t and masks m_t
# - Rauch–Tung–Striebel (RTS) smoother
# - Utilities for reconstruction and k-step forecasting
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    # Clip inputs to avoid overflow in exp for large |x|
    x_clipped = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def _symmetrize(M: np.ndarray) -> np.ndarray:
    """Force exact symmetry (helps EKF/RTS numerical stability)."""
    return 0.5 * (M + M.T)

def _stable_mean_ignore_zeros_from_nanfill(x_filled: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    """
    If x_filled was created by nan_to_num(nan=0), using np.mean will bias downward.
    Compute mean using obs_mask (1=observed, 0=missing).
    Shapes:
    x_filled: (T,D)
    obs_mask: (T,D)
    Returns:
    x_mean: (D,)
    """
    num = (x_filled * obs_mask).sum(axis=0)
    den = obs_mask.sum(axis=0) + 1e-6
    return (num / den).astype(np.float32)

# ------------------------------------------------------------
# Parameter container
# ------------------------------------------------------------

@dataclass
class MNARParams:
    """
    Container for MNAR LDS parameters.

    Dimensions
    ----------
    - K : latent state dimension
    - D : number of detectors / observed variables
    """
    A: np.ndarray          # (K, K)  state transition
    Q: np.ndarray          # (K, K)  process noise covariance
    C: np.ndarray          # (D, K)  emission matrix
    R: np.ndarray          # (D, D)  observation noise covariance for speeds
    mu0: np.ndarray        # (K,)    initial mean
    Sigma0: np.ndarray     # (K, K)  initial covariance
    # MNAR mask model:
    #   logit_{t,d} = phi_d^T z_t + beta_d^T time_t + b_d
    phi: np.ndarray        # (D, K)       weights on latent state
    beta_time: np.ndarray  # (D, F_time)  weights on exogenous time features
    b: np.ndarray          # (D,)         per-detector intercept (base missing rate)

    @property
    def K(self) -> int:
        # Latent state dimension K (dimensionality of z_t)
        return self.A.shape[0]

    @property
    def D(self) -> int:
        # Observation dimension D (number of detectors / observed speeds)
        return self.C.shape[0]

    @staticmethod
    def init_random(K: int, D: int, seed: int = 0, F_time: int = 6):
        """
        Simple random initialization for debugging / experimentation.
        Replace with learned params for real experiments.
        """
        rng = np.random.default_rng(seed)

        # Randomly stable-ish A (shrink towards identity)
        A = np.eye(K) + 0.05 * rng.standard_normal((K, K))
        # Process noise (diagonal for simplicity)
        Q_diag = 0.1 * np.ones(K, dtype=float)
        Q = np.diag(Q_diag)

        # Emission matrix C: map K-dim latent state to D detectors
        C = 0.1 * rng.standard_normal((D, K))

        # Observation noise for speeds (diagonal)
        R_diag = 4.0 * np.ones(D, dtype=float)  # e.g. ~2 m/s std
        R = np.diag(R_diag)

        # Initial state prior
        mu0 = np.zeros(K, dtype=float)
        Sigma0 = np.eye(K, dtype=float)

        # MNAR mask model params
        phi = 0.1 * rng.standard_normal((D, K))
        # Default time-feature dimension unknown at init; start with 0 columns.
        beta_time = np.zeros((D, F_time), dtype=float)
        b = (-2.944 * np.ones(D, dtype=float))

        return MNARParams(A=A, Q=Q, C=C, R=R, mu0=mu0, Sigma0=Sigma0, phi=phi, beta_time=beta_time, b=b)


# ------------------------------------------------------------
# Core MNAR-aware EKF
# ------------------------------------------------------------

class MNARBlackoutLDS:
    """
    MNAR-aware linear dynamical system for traffic blackouts.

    The model matches the LaTeX description:

        z_t | z_{t-1} ~ N(A z_{t-1}, Q)
        x_t | z_t     ~ N(C z_t, R)
        m_{t,d} | z_t ~ Bernoulli( sigma( phi_d^T z_t ) )

    where:
        - x_t in R^D : detector speeds
        - m_t in {0,1}^D, with 1 = missing, 0 = observed

    Inference is done with an EKF-style update that uses:
        - observed speeds x_t (only where m_t == 0),
        - full masks m_t as an additional "pseudo-observation" block.
    """

    def __init__(
        self,
        params: MNARParams,
        use_missingness_obs: bool = True,
        missingness_var_mode: str = "clipped_moment",
        missingness_var_const: float = 0.05,
        missingness_var_clip: tuple[float, float] = (1e-2, 0.25),
        solve_jitter: float = 1e-6,
        missingness_max_obs0: int = 64,
        missingness_max_total: Optional[int] = 256,
    ):
        self.params = params

        # If False, EKF uses ONLY observed speeds (MAR-style inference).
        self.use_missingness_obs = bool(use_missingness_obs)
        # Controls pseudo-Gaussian variance for Bernoulli mask observations.
        #  - "moment": var = pi*(1-pi)
        #  - "constant": var = missingness_var_const
        #  - "clipped_moment": clip(pi*(1-pi), [lo, hi])
        self.missingness_var_mode = str(missingness_var_mode)
        self.missingness_var_const = float(missingness_var_const)
        self.missingness_var_clip = tuple(missingness_var_clip)
        self._solve_jitter = float(solve_jitter)

        # include all missing detectors + up to missingness_max_obs0 observed zeros
        self.missingness_max_obs0 = int(missingness_max_obs0)
        # optional hard cap on total missingness pseudo-observations per time step
        self.missingness_max_total = None if missingness_max_total is None else int(missingness_max_total)

    # --------------------------------------------------------
    # Forward pass: MNAR-aware extended Kalman filter
    # --------------------------------------------------------

    def ekf_forward(
        self,
        x_t: np.ndarray,
        m_t: np.ndarray,
        X_time: Optional[np.ndarray] = None,
        use_missingness_obs: Optional[bool] = None,
        missingness_var_mode: Optional[str] = None,
        missingness_var_const: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run MNAR-aware EKF over the entire sequence.

        Parameters
        ----------
        x_t : np.ndarray, shape (T, D)
            Speed panel. NaNs are allowed; they should be consistent
            with m_t (i.e., NaN when m_t[t, d] == 1).
        m_t : np.ndarray, shape (T, D), dtype in {0,1}
            Missingness masks (1 = missing, 0 = observed).

        Returns
        -------
        results : dict of np.ndarray
            - 'mu_pred'   : (T, K)  predicted means   E[z_t | y_{1:t-1}]
            - 'Sigma_pred': (T, K, K) predicted covs
            - 'mu_filt'   : (T, K)  filtered means    E[z_t | y_{1:t}]
            - 'Sigma_filt': (T, K, K) filtered covs
        """
        params = self.params
        A, Q, C, R, mu0, Sigma0, phi, beta_time, b = (
            params.A,
            params.Q,
            params.C,
            params.R,
            params.mu0,
            params.Sigma0,
            params.phi,
            params.beta_time,
            params.b,
        )

        T, D = x_t.shape
        K = params.K

        # If no time features provided, use zeros (equivalent to no exogenous effect).
        if X_time is None:
            X_time = np.zeros((T, 0), dtype=float)
        if X_time.ndim != 2 or X_time.shape[0] != T:
            raise ValueError(f"X_time must be (T,F). Got {X_time.shape} for T={T}.")
        F_time = X_time.shape[1]
        # Allow beta_time to be lazily initialized to correct shape.
        if beta_time.shape != (D, F_time):
            if beta_time.size == 0 and F_time > 0:
                beta_time = np.zeros((D, F_time), dtype=float)
                self.params.beta_time = beta_time
            else:
                raise ValueError(f"beta_time shape {beta_time.shape} does not match (D,F)={(D,F_time)}")

        # Allocate arrays for predicted / filtered statistics
        mu_pred = np.zeros((T, K), dtype=float)
        Sigma_pred = np.zeros((T, K, K), dtype=float)
        mu_filt = np.zeros((T, K), dtype=float)
        Sigma_filt = np.zeros((T, K, K), dtype=float)

        # Start from prior at t = 0
        mu_prev = mu0.copy()
        Sigma_prev = Sigma0.copy()

        I_K = np.eye(K, dtype=float)

        # deterministic RNGs (don’t recreate inside the time loop)
        rng_obs0 = np.random.default_rng(0)
        rng_cap  = np.random.default_rng(1)

        for t in range(T):
            # ------------------------------------------------
            # 1) Predict step (standard LDS)
            # ------------------------------------------------
            # IMPORTANT: at t=0, prior is (mu0, Sigma0) for z_0.
            # Only start applying dynamics from t>=1.
            if t == 0:
                mu_t_pred = mu_prev
                Sigma_t_pred = Sigma_prev
            else:
                mu_t_pred = A @ mu_prev
                Sigma_t_pred = A @ Sigma_prev @ A.T + Q

            # Save prediction before seeing time t data
            mu_pred[t] = mu_t_pred
            Sigma_pred[t] = Sigma_t_pred

            # ------------------------------------------------
            # 2) Build observation blocks at time t
            # ------------------------------------------------
            mask_row = m_t[t]  # shape (D,)
            # Indices of observed speeds (m = 0)
            # Also guard against accidental NaNs even when mask says observed
            observed_idx = np.where((mask_row == 0) & (~np.isnan(x_t[t])))[0]
            has_speeds = observed_idx.size > 0

            # --- Speed block (only if some speeds observed) ---
            if has_speeds:
                # Observed speed values y_t^{(x)}
                y_x = x_t[t, observed_idx].astype(float)  # (|O_t|,)

                # Observation model h^{(x)}(z) = C_x z
                C_x = C[observed_idx, :]                 # (|O_t|, K)
                h_x = C_x @ mu_t_pred                    # (|O_t|,)

                # Jacobian for speed block is just C_x (linear)
                J_x = C_x

                # Submatrix of R restricted to observed detectors
                R_x = R[np.ix_(observed_idx, observed_idx)]  # (|O_t|, |O_t|)
            else:
                # Placeholders (unused if has_speeds is False)
                y_x = None
                h_x = None
                J_x = None
                R_x = None

            # --- Missingness block (always present) ---
            # Decide whether to use missingness pseudo-observations this pass
            use_m = self.use_missingness_obs if use_missingness_obs is None else bool(use_missingness_obs)
            mode = self.missingness_var_mode if missingness_var_mode is None else str(missingness_var_mode)
            var_const = self.missingness_var_const if missingness_var_const is None else float(missingness_var_const)

            if use_m:
                # We model probability of m_{t,d} = 1 (missing) as:
                #   pi_{t,d} = sigma( phi_d^T mu_{t|t-1} )
                # logit_{t,d} = phi_d^T z_t + beta_d^T time_t + b_d
                u_m = (phi @ mu_t_pred)                      # (D,)
                if F_time > 0:
                    u_m = u_m + (beta_time @ X_time[t])      # (D,)
                u_m = u_m + b                                # (D,)
                pi = _sigmoid(u_m)                           # (D,)
                # Jacobian (local linearization / EKF):
                #   d pi / d z = pi*(1-pi) * phi
                g = (pi * (1.0 - pi))[:, None] * phi         # (D, K)

                # ------------------------------------------------------------
                # Subsample missingness pseudo-observations to avoid O(D^3)
                # Include:
                #   - all missing indices (m==1)
                #   - a capped subset of observed zeros (m==0) for extra signal
                #   - optional hard cap on total to keep EKF stable/fast
                # ------------------------------------------------------------
                miss_idx = np.where(mask_row == 1)[0]
                obs0_idx = np.where(mask_row == 0)[0]
 
                # cap observed-zero indices
                cap_obs0 = self.missingness_max_obs0
                if cap_obs0 is not None and obs0_idx.size > cap_obs0:
                    obs0_idx = rng_obs0.choice(obs0_idx, size=cap_obs0, replace=False)
 
                idx_m = np.unique(np.concatenate([miss_idx, obs0_idx], axis=0))
 
                # optional hard cap on total size
                if self.missingness_max_total is not None and idx_m.size > self.missingness_max_total:
                    idx_m = rng_cap.choice(idx_m, size=self.missingness_max_total, replace=False)
 
                # Pseudo-Gaussian observation for masks (SLICED):
                y_m = mask_row[idx_m].astype(float)          # (M_m,)
                h_m = pi[idx_m]                              # (M_m,)
                J_m = g[idx_m, :]                            # (M_m, K)

                # Variance choice (ablation-ready)
                if mode == "moment":
                    pi_s = pi[idx_m]
                    var_m = pi_s * (1.0 - pi_s)
                elif mode == "constant":
                    var_m = np.full(idx_m.size, var_const, dtype=float)
                elif mode == "clipped_moment":
                    lo, hi = self.missingness_var_clip
                    pi_s = pi[idx_m]
                    var_m = np.clip(pi_s * (1.0 - pi_s), lo, hi)
                else:
                    raise ValueError(f"Unknown missingness_var_mode='{mode}'")

                var_m = var_m + 1e-6
                S_m = np.diag(var_m)
            else:
                # If not using missingness observations, we simply skip that block.
                y_m = h_m = J_m = S_m = None

            # ------------------------------------------------
            # 3) Combine blocks and run EKF update
            # ------------------------------------------------
            if has_speeds:
                if use_m:
                    # Stack observed speeds and missingness
                    y = np.concatenate([y_x, y_m], axis=0)
                    h = np.concatenate([h_x, h_m], axis=0)
                    J = np.vstack([J_x, J_m])

                    # Block-diagonal covariance: diag(R_x, S_m)
                    top_left = R_x
                    M_m = y_m.shape[0]
                    top_right = np.zeros((R_x.shape[0], M_m), dtype=float)
                    bottom_left = np.zeros((M_m, R_x.shape[0]), dtype=float)
                    bottom_right = S_m
                    R_t = np.block([[top_left, top_right],[bottom_left, bottom_right]])
                else:
                    # Speeds-only update (MAR inference)
                    y = y_x
                    h = h_x
                    J = J_x
                    R_t = R_x
            else:
                if use_m:
                    # No speeds: missingness-only update
                    y = y_m
                    h = h_m
                    J = J_m
                    R_t = S_m
                else:
                    # No observation at all -> posterior = prediction
                    mu_t_filt = mu_t_pred
                    Sigma_t_filt = Sigma_t_pred
                    mu_filt[t] = mu_t_filt
                    Sigma_filt[t] = Sigma_t_filt
                    mu_prev = mu_t_filt
                    Sigma_prev = Sigma_t_filt
                    continue

            # Innovation covariance: S_y = J Σ J^T + R
            S_y = J @ Sigma_t_pred @ J.T + R_t
            # Stabilize S_y a touch
            S_y = _symmetrize(S_y) + self._solve_jitter * np.eye(S_y.shape[0], dtype=float)

            # Kalman gain (solve, not explicit inverse):
            #   K_t = Σ J^T S_y^{-1}
            # Compute K_t via: K_t = (solve(S_y, (Σ J^T)^T))^T
            SJt = (Sigma_t_pred @ J.T)               # (K, M)
            K_t = np.linalg.solve(S_y, SJt.T).T      # (K, M)

            # Innovation: (y - h(z_pred))
            innov = y - h

            # Filtered state mean and covariance
            mu_t_filt = mu_t_pred + K_t @ innov
            # Joseph form covariance update (numerically safer):
            #   Σ = (I-KJ) Σ (I-KJ)^T + K R K^T
            IKJ = (I_K - K_t @ J)
            Sigma_t_filt = IKJ @ Sigma_t_pred @ IKJ.T + K_t @ R_t @ K_t.T
            Sigma_t_filt = _symmetrize(Sigma_t_filt)

            # Save filtered posterior
            mu_filt[t] = mu_t_filt
            Sigma_filt[t] = Sigma_t_filt

            # Prepare for next time step
            mu_prev = mu_t_filt
            Sigma_prev = Sigma_t_filt

        return {
            "mu_pred": mu_pred,
            "Sigma_pred": Sigma_pred,
            "mu_filt": mu_filt,
            "Sigma_filt": Sigma_filt,
        }

    # --------------------------------------------------------
    # Backward pass: Rauch–Tung–Striebel smoother
    # --------------------------------------------------------

    def rts_smoother(
        self,
        ekf_results: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        RTS smoother for the MNAR LDS.

        This uses the standard LDS RTS equations, applied to the
        EKF-filtered sequence (we ignore higher-order corrections and
        treat the linearization as fixed).

        Parameters
        ----------
        ekf_results : dict
            Output of ekf_forward():
            - 'mu_pred', 'Sigma_pred', 'mu_filt', 'Sigma_filt'.

        Returns
        -------
        results : dict of np.ndarray
            - 'mu_smooth'   : (T, K)   smoothed means
            - 'Sigma_smooth': (T, K, K) smoothed covariances
        """
        A = self.params.A

        mu_pred = ekf_results["mu_pred"]      # (T, K)
        Sigma_pred = ekf_results["Sigma_pred"]  # (T, K, K)
        mu_filt = ekf_results["mu_filt"]      # (T, K)
        Sigma_filt = ekf_results["Sigma_filt"]  # (T, K, K)

        T, K = mu_filt.shape

        # Initialize smoothed arrays with filtered values
        mu_smooth = mu_filt.copy()
        Sigma_smooth = Sigma_filt.copy()

        # Iterate backward: t = T-2, ..., 0
        for t in range(T - 2, -1, -1):
            # Smoother gain:
            #   F_t = Σ_{t|t} A^T (Σ_{t+1|t})^{-1}
            Sigma_f = Sigma_filt[t]            # (K, K)
            Sigma_pred_next_raw = Sigma_pred[t + 1]  # (K, K)
            Sigma_pred_next_stab = _symmetrize(Sigma_pred_next_raw) + self._solve_jitter * np.eye(K, dtype=float)
            F_t = np.linalg.solve(Sigma_pred_next_stab, (Sigma_f @ A.T).T).T

            # Mean update:
            #   μ_{t|T} = μ_{t|t} + F_t (μ_{t+1|T} - μ_{t+1|t})
            mu_smooth[t] = (
                mu_filt[t]
                + F_t @ (mu_smooth[t + 1] - mu_pred[t + 1])
            )

            # Covariance update:
            #   Σ_{t|T} = Σ_{t|t} + F_t (Σ_{t+1|T} - Σ_{t+1|t}) F_t^T
            Sigma_smooth[t] = (
                Sigma_f
                + F_t @ (Sigma_smooth[t + 1] - Sigma_pred_next_raw) @ F_t.T
            )
            Sigma_smooth[t] = _symmetrize(Sigma_smooth[t])

        return {
            "mu_smooth": mu_smooth,
            "Sigma_smooth": Sigma_smooth,
        }

    # --------------------------------------------------------
    # EM training for MNAR LDS
    # --------------------------------------------------------

    def em_train(
        self,
        x_t: np.ndarray,
        m_t: np.ndarray,
        X_time: Optional[np.ndarray] = None,
        num_iters: int = 5,
        update_phi: bool = True,
        phi_steps: int = 50,
        phi_lr: float = 3e-3,
        phi_l2: float = 1e-3,
        neg_per_pos: int = 3,
        grad_clip: float = 5.0,
        anneal_missingness_var: bool = True,
        verbose: bool = True,
        convergence_tol: Optional[float] = None,
        use_missingness_obs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run EM (approximate) for the MNAR LDS on a single sequence.

        This alternates:
            - E-step: MNAR-aware EKF + RTS smoother
            - M-step: closed-form updates for (mu0, Sigma0, A, Q, C, R)
                      + gradient-ascent updates for phi

        Notes
        -----
        * We use the smoothed posterior (mu_smooth, Sigma_smooth) as
          the approximate q(z_{0:T}).
        * Cross-covariances E[z_t z_{t-1}^T] for A,Q are approximated
          using outer products of smoothed means:
              S_{t,t-1} ≈ mu_t mu_{t-1}^T
          This is not exact RTS EM, but is usually a reasonable
          approximation and keeps the code simple.
        * Phi is updated by logistic regression on smoothed states:
              m_{t,d} ~ Bernoulli( sigma( phi_d^T mu_{t|T} ) )
          using a few gradient steps per EM iteration.

        Parameters
        ----------
        x_t : np.ndarray, shape (T, D)
            Speed panel (NaNs allowed, must match m_t).
        m_t : np.ndarray, shape (T, D)
            Missingness indicators (1 = missing, 0 = observed).
        num_iters : int
            Number of EM iterations.
        update_phi : bool
            If True, update phi each iteration using gradient ascent.
        phi_steps : int
            Number of gradient steps for each detector per EM iter.
        phi_lr : float
            Learning rate for phi updates.
        verbose : bool
            If True, print progress each iteration.
        convergence_tol : Optional[float]
            If not None, perform early stopping when the maximum
            relative change in {A, C, phi} between successive
            iterations drops below this threshold.

        Returns
        -------
        history : dict
            Simple history with parameter snapshots per iteration.
        """
        T, D = x_t.shape
        K = self.params.K

        history = {
            "A": [],
            "Q": [],
            "C": [],
            "R_diag": [],
            "phi": [],
        }

        for it in range(num_iters):
            if verbose:
                print(f"\n=== EM iteration {it + 1}/{num_iters} ===")

            # -------------------------------
            # E-step: EKF + RTS smoother
            # -------------------------------
            # Anneal: start stable, then make masks more influential
            if anneal_missingness_var:
                # e.g. 0.25 -> 0.10 -> 0.05 -> 0.02 (floor)
                var_const = max(0.02, 0.25 * (0.6 ** it))
            else:
                var_const = self.missingness_var_const

            ekf_res = self.ekf_forward(
                x_t=x_t,
                m_t=m_t,
                X_time=X_time,
                use_missingness_obs=use_missingness_obs,
                missingness_var_const=var_const,
            )
            smooth_res = self.rts_smoother(ekf_res)

            mu_smooth = smooth_res["mu_smooth"]        # (T, K)
            Sigma_smooth = smooth_res["Sigma_smooth"]  # (T, K, K)

            # S_t = E[z_t z_t^T] = Σ_{t|T} + μ_{t|T} μ_{t|T}^T
            S_t = Sigma_smooth + np.einsum("ti,tj->tij", mu_smooth, mu_smooth)

            # ------------------------------------------------
            # M-step: Initial state (mu0, Sigma0)
            # ------------------------------------------------
            mu0_new = mu_smooth[0].copy()         # (K,)
            Sigma0_new = Sigma_smooth[0].copy()   # (K, K)

            # ------------------------------------------------
            # M-step: Dynamics (A, Q) using approximate stats
            # ------------------------------------------------
            # Sum over t = 1..T-1 (since we use z_{t-1})
            S_t_sum = np.sum(S_t[1:], axis=0)          # sum S_t, t>=1
            S_tminus1_sum = np.sum(S_t[:-1], axis=0)   # sum S_{t-1}, t=0..T-2

            # Approximate cross S_{t,t-1} ~ mu_t mu_{t-1}^T
            S_cross_sum = np.zeros((K, K), dtype=float)
            for t in range(1, T):
                S_cross_sum += np.outer(mu_smooth[t], mu_smooth[t - 1])

            # A_new = (sum_t S_{t,t-1}) (sum_t S_{t-1})^{-1}
            A_new = S_cross_sum @ np.linalg.inv(S_tminus1_sum + 1e-8 * np.eye(K))

            # --- Regularization: shrink A toward identity to keep dynamics stable ---
            lam_A = 0.05  # e.g., 5% pull toward I; tune if needed
            A_new = (1.0 - lam_A) * A_new + lam_A * np.eye(K)

            # Q_new from:
            #   Q = 1/(T-1) sum_t [
            #         S_t
            #       - A S_{t,t-1}^T
            #       - S_{t,t-1} A^T
            #       + A S_{t-1} A^T
            #     ]
            Q_accum = np.zeros((K, K), dtype=float)
            for t in range(1, T):
                S_curr = S_t[t]
                S_prev = S_t[t - 1]

                # Approximate cross-cov S_{t,t-1}
                S_cross = np.outer(mu_smooth[t], mu_smooth[t - 1])

                term = (
                    S_curr
                    - A_new @ S_cross.T
                    - S_cross @ A_new.T
                    + A_new @ S_prev @ A_new.T
                )
                Q_accum += term

            Q_new = Q_accum / (T - 1)
            
            # Symmetrize and add jitter for numerical stability
            Q_new = 0.5 * (Q_new + Q_new.T)
            Q_new += 1e-6 * np.eye(K)

            # --- Regularization: shrink Q toward an isotropic prior and cap its scale ---
            lam_Q = 0.3          # how much to pull Q toward the prior; tune if needed
            Q_prior = 0.1 * np.eye(K)  # prior process noise level
            Q_new = (1.0 - lam_Q) * Q_new + lam_Q * Q_prior

            # Cap the overall scale of Q to avoid exploding dynamics
            max_trace = 30.0    # maximum allowed trace of Q; tune if needed
            tr_Q = float(np.trace(Q_new))
            if tr_Q > max_trace:
                Q_new *= max_trace / tr_Q

            # ------------------------------------------------
            # M-step: Emissions (C, R) with masking
            # ------------------------------------------------
            C_new = np.zeros_like(self.params.C)       # (D, K)
            R_new = np.zeros_like(self.params.R)       # (D, D)

            for d in range(D):
                # Times where detector d is observed
                obs_idx = np.where((m_t[:, d] == 0) & (~np.isnan(x_t[:, d])))[0]
                if obs_idx.size == 0:
                    # No data for this detector; keep old params
                    C_new[d] = self.params.C[d]
                    R_new[d, d] = self.params.R[d, d]
                    continue
                R_new = np.diag(np.diag(R_new))

                # Vectorized sufficient statistics for C_d and R_dd
                # Shapes:
                #   S_obs  : (#obs, K, K)
                #   mu_obs : (#obs, K)
                #   x_obs  : (#obs,)
                S_obs = S_t[obs_idx]                # (N_obs, K, K)
                mu_obs = mu_smooth[obs_idx]         # (N_obs, K)
                x_obs = x_t[obs_idx, d].astype(float)  # (N_obs,)

                # Denominator: sum_t S_t
                S_sum_d = S_obs.sum(axis=0)         # (K, K)

                # Numerator: sum_t x_{t,d} mu_t^T
                # (N_obs, 1) * (N_obs, K) -> (N_obs, K) -> sum over obs
                num_d = (x_obs[:, None] * mu_obs).sum(axis=0)  # (K,)

                # C_d = num_d (sum_t S_t)^{-1}
                C_d_row = num_d @ np.linalg.inv(S_sum_d + 1e-8 * np.eye(K))
                C_new[d, :] = C_d_row

                # Now compute R_dd
                # For each t:
                #   x_{t,d}^2
                #   - 2 x_{t,d} (C_d mu_t)
                #   + C_d S_t C_d^T
                # Vectorized:
                #   preds  = C_d mu_t  for all obs
                #   C_S_Ct = C_d S_t C_d^T  for all obs

                # preds = C_d mu_t  -> (N_obs,)
                preds = mu_obs @ C_d_row            # (N_obs,)

                # C_S_C = C_d S_t C_d^T per timestep t
                # einsum index explanation:
                #   i   : latent index (for C_d_row left)
                #   t,i,j : S_obs[t, i, j]
                #   j   : latent index (for C_d_row right)
                # Result: (t)
                C_S_C = np.einsum("i,tij,j->t", C_d_row, S_obs, C_d_row)  # (N_obs,)

                err = x_obs**2 - 2.0 * x_obs * preds + C_S_C             # (N_obs,)
                R_dd = float(err.mean())
                R_dd = max(R_dd, 1e-6)    # Ensure non-negative + jitter
                R_new[d, d] = R_dd

            # ------------------------------------------------
            # M-step: Missingness parameters phi (logistic)
            # ------------------------------------------------
            phi_new = self.params.phi.copy()  # (D, K)
            # Handle time features for mask model
            if X_time is None:
                X_time_local = np.zeros((T, 0), dtype=float)
            else:
                X_time_local = X_time.astype(float)
            F_time = X_time_local.shape[1]
            beta_new = self.params.beta_time.copy()
            b_new = self.params.b.copy()
            if beta_new.shape != (D, F_time):
                if beta_new.size == 0 and F_time > 0:
                    beta_new = np.zeros((D, F_time), dtype=float)
                elif F_time == 0:
                    beta_new = np.zeros((D, 0), dtype=float)
                else:
                    raise ValueError(f"beta_time shape {beta_new.shape} != {(D,F_time)}")
            if update_phi:
                # Design: [Z | X_time | 1]
                Z = mu_smooth.astype(float)  # (T, K)

                for d in range(D):
                    y = m_t[:, d].astype(float)  # (T,)
                    pos_idx = np.where(y == 1.0)[0]
                    neg_idx = np.where(y == 0.0)[0]
                    if pos_idx.size == 0 or neg_idx.size == 0:
                        continue

                    # BALANCE: sample negatives proportional to positives
                    rs = np.random.default_rng(1000 + d + 17 * it)
                    n_pos = min(pos_idx.size, 20000)
                    pos_s = rs.choice(pos_idx, size=n_pos, replace=False)
                    n_neg = min(neg_idx.size, neg_per_pos * n_pos)
                    neg_s = rs.choice(neg_idx, size=n_neg, replace=False)
                    idx = np.concatenate([pos_s, neg_s])
                    rs.shuffle(idx)

                    Zb = Z[idx]                         # (N, K)
                    Tb = X_time_local[idx]              # (N, F_time)
                    yb = y[idx]                         # (N,)

                    phi_d = phi_new[d].copy()           # (K,)
                    beta_d = beta_new[d].copy()         # (F_time,)
                    b_d = float(b_new[d])               # scalar

                    # Simple clipped SGD on balanced set
                    N = float(yb.size)
                    for _ in range(phi_steps):
                        logits = (Zb @ phi_d)
                        if F_time > 0:
                            logits = logits + (Tb @ beta_d)
                        logits = logits + b_d

                        p = _sigmoid(logits)            # (N,)
                        err = (yb - p)                  # (N,)

                        # Gradients (ASCENT on log-likelihood)
                        g_phi = (Zb.T @ err) / N - phi_l2 * phi_d
                        if F_time > 0:
                            g_beta = (Tb.T @ err) / N - phi_l2 * beta_d
                        else:
                            g_beta = None
                        g_b = float(np.sum(err) / N)

                        # Clip
                        g_phi = np.clip(g_phi, -grad_clip, grad_clip)
                        if g_beta is not None:
                            g_beta = np.clip(g_beta, -grad_clip, grad_clip)
                        g_b = float(np.clip(g_b, -grad_clip, grad_clip))

                        phi_d += phi_lr * g_phi
                        if g_beta is not None:
                            beta_d += phi_lr * g_beta
                        b_d += phi_lr * g_b

                    phi_new[d] = phi_d
                    beta_new[d] = beta_d
                    b_new[d] = b_d

            # ------------------------------------------------
            # Compute relative parameter change (for early stop)
            # ------------------------------------------------
            max_rel_change = None
            if convergence_tol is not None and it > 0:
                # Helper to compute relative Frobenius change
                def _rel_change(new: np.ndarray, old: np.ndarray) -> float:
                    num = np.linalg.norm(new - old)
                    denom = np.linalg.norm(old) + 1e-8
                    return num / denom

                prev_params = self.params
                delta_A = _rel_change(A_new, prev_params.A)
                delta_C = _rel_change(C_new, prev_params.C)
                delta_phi = _rel_change(phi_new, prev_params.phi)
                max_rel_change = max(delta_A, delta_C, delta_phi)

            # ------------------------------------------------
            # Update the parameter container
            # ------------------------------------------------
            self.params = MNARParams(
                A=A_new,
                Q=Q_new,
                C=C_new,
                R=R_new,
                mu0=mu0_new,
                Sigma0=Sigma0_new,
                phi=phi_new,
                beta_time=beta_new,
                b=b_new,
            )

            # Log history (e.g., for debugging)
            history["A"].append(A_new.copy())
            history["Q"].append(Q_new.copy())
            history["C"].append(C_new.copy())
            history["R_diag"].append(np.diag(R_new).copy())
            history["phi"].append(phi_new.copy())

            if verbose:
                mean_R = float(np.mean(np.diag(R_new)))
                print(f"  A norm: {np.linalg.norm(A_new):.3f}")
                print(f"  Q trace: {np.trace(Q_new):.3f}")
                print(f"  mean diag(R): {mean_R:.3f}")
                if max_rel_change is not None:
                    print(f"  max relative param change: {max_rel_change:.3e}")

            # ------------------------------------------------
            # Early stopping condition
            # ------------------------------------------------
            if (
                convergence_tol is not None
                and max_rel_change is not None
                and max_rel_change < convergence_tol
            ):
                if verbose:
                    print(
                        f"  Early stopping at iter {it + 1} "
                        f"(Δ={max_rel_change:.3e} < tol={convergence_tol:.1e})"
                    )
                break

        return history

    # --------------------------------------------------------
    # Calculation of log-likelihood
    # --------------------------------------------------------
    def compute_log_likelihood(
        self,
        x_t: np.ndarray,
        m_t: np.ndarray,
        ekf_results: Dict[str, np.ndarray],
        include_missingness: bool = False,
        X_time: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute log-likelihood of observed data under the MNAR/MAR LDS.

        This uses the EKF forward pass to compute the one-step-ahead
        predictive log-likelihood at each time step, then sums over t.

        Parameters
        ----------
        x_t : np.ndarray, shape (T, D)
            Speed panel (NaNs allowed, must match m_t).
        m_t : np.ndarray, shape (T, D)
            Missingness indicators (1 = missing, 0 = observed).

        Returns
        -------
        log_likelihood : float
            Total log-likelihood of observed data.
        """
        mu_pred = ekf_results["mu_pred"]          # (T, K)
        Sigma_pred = ekf_results["Sigma_pred"]    # (T, K, K)

        params = self.params
        A, Q, C, R, phi, beta_time, b = (
            params.A, params.Q, params.C, params.R, params.phi, params.beta_time, params.b
        )

        if X_time is None:
            X_time = np.zeros((T, 0), dtype=float)
        F_time = X_time.shape[1]
        if beta_time.shape != (params.D, F_time):
            if beta_time.size == 0 and F_time > 0:
                beta_time = np.zeros((params.D, F_time), dtype=float)
            elif F_time != 0:
                raise ValueError("beta_time shape mismatch in LL.")

        T, _ = x_t.shape
        log_likelihood = 0.0

        for t in range(T):
            mu_t = mu_pred[t]                      # (K,)
            Sigma_t = Sigma_pred[t]                # (K, K)

            # --- Speed block ---
            mask_row = m_t[t]                   # (D,)
            observed_idx = np.where((mask_row == 0) & np.isfinite(x_t[t]))[0]
            if observed_idx.size > 0:
                y_x = x_t[t, observed_idx].astype(float)  # (|O_t|,)
                C_x = C[observed_idx, :]                  # (|O_t|, K)
                R_x = R[np.ix_(observed_idx, observed_idx)]  # (|O_t|, |O_t|)

                h_x = C_x @ mu_t                           # (|O_t|,)
                S_x = C_x @ Sigma_t @ C_x.T + R_x         # (|O_t|, |O_t|)

                diff_x = y_x - h_x
                try:                         # (|O_t|,) Innovation
                    S_x = _symmetrize(S_x) + self._solve_jitter * np.eye(S_x.shape[0], dtype=float)
                    quad = diff_x.T @ np.linalg.solve(S_x, diff_x)
                    sign, logdet = np.linalg.slogdet(S_x)
                    if sign <= 0:
                        ll_x = -np.inf
                    else:
                        n = S_x.shape[0]
                        ll_x = -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))
                except np.linalg.LinAlgError:
                    ll_x = -np.inf
                    
                log_likelihood += ll_x

            # --- Missingness block (optional LL term) ---
            if include_missingness:
                # Bernoulli log-likelihood under pi_d = sigmoid(phi_d^T z)
                logits = (phi @ mu_t)
                if F_time > 0:
                    logits = logits + (beta_time @ X_time[t])
                logits = logits + b
                pi = _sigmoid(logits)                 # (D,)
                pi = np.clip(pi, 1e-8, 1.0 - 1e-8)
                y_m = m_t[t].astype(float)            # (D,)
                ll_m = np.sum(y_m * np.log(pi) + (1.0 - y_m) * np.log(1.0 - pi))
                log_likelihood += float(ll_m)

        return log_likelihood

    # --------------------------------------------------------
    # Reconstruction & forecasting utilities
    # --------------------------------------------------------

    def reconstruct_from_smoother(
        self,
        mu_smooth: np.ndarray,
        Sigma_smooth: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct x_t from smoothed latent states.

        Uses:
            x_hat_t = C * mu_{t|T}

        Parameters
        ----------
        mu_smooth : np.ndarray, shape (T, K)
            Smoothed latent means from rts_smoother().

        Returns
        -------
        x_hat : np.ndarray, shape (T, D)
            Reconstructed speed panel.
        cov : np.ndarray, shape (T, D, D)
            Reconstructed covariance matrices.
        """
        C = self.params.C
        # Matrix multiply for all T at once:
        # (T, K) @ (K, D)^T  => (T, D)
        x_hat = mu_smooth @ C.T

        cov = np.einsum("dk,tkm,em->tde", C, Sigma_smooth, C) + self.params.R[None, :, :]

        return x_hat, cov

    def k_step_forecast(
        self,
        mu_filt: np.ndarray,
        Sigma_filt: np.ndarray,
        start_idx: int,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        k-step-ahead forecast starting from time 'start_idx'.

        This corresponds to forecasting from the end of a blackout:
        you condition on data up to time 'start_idx', then propagate
        forward using the dynamics.

        Parameters
        ----------
        mu_filt : np.ndarray, shape (T, K)
            Filtered means from ekf_forward().
        Sigma_filt : np.ndarray, shape (T, K, K)
            Filtered covariances from ekf_forward().
        start_idx : int
            Index 'b' where the blackout ends. We forecast from t = b.
        k : int
            Horizon length (in steps), e.g. 1, 3, or 6.

        Returns
        -------
        mean_x : np.ndarray, shape (D,)
            Forecast mean x_{b+k}.
        cov_x : np.ndarray, shape (D, D)
            Forecast covariance of x_{b+k}.
        """
        A, Q, C, R = self.params.A, self.params.Q, self.params.C, self.params.R

        # Start from filtered posterior at time b
        mu = mu_filt[start_idx].copy()        # (K,)
        Sigma = Sigma_filt[start_idx].copy()  # (K, K)

        # Propagate state forward k steps with dynamics
        for _ in range(k):
            mu = A @ mu
            Sigma = A @ Sigma @ A.T + Q

        # Map to observation space:
        #   x_{b+k} ~ N(C mu, C Σ C^T + R)
        mean_x = C @ mu                          # (D,)
        cov_x = C @ Sigma @ C.T + R              # (D, D)

        return mean_x, cov_x

