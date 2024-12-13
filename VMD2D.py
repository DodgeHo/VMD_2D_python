import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def VMD2D(
    signal: np.ndarray, 
    alpha: float, tau: float, K: int, DC: bool, init: int, tol: float, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #
    # Input and Parameters:
    # ---------------------
    #  signal  - the space domain signal (2D) to be decomposed
    #  alpha   - the balancing parameter of the data-fidelity constraint
    #  tau     - time-step of the dual ascent ( pick 0 for noise-slack )
    #  K       - the number of modes to be recovered
    #  DC      - true if the first mode is put and kept at DC (0-freq)
    #  init    - 0 = all omegas start at 0
    #            1 = all omegas start initialized randomly
    #  tol     - tolerance of convergence criterion; typically around 1e-7
    #
    #  Output:
    #  -------
    #  u       - the collection of decomposed modes
    #  u_hat   - spectra of the modes
    #  omega   - estimated mode center-frequencies
    #

    # Validate input parameters
    if not isinstance(signal, np.ndarray) or signal.ndim != 2:
        raise ValueError("Signal must be a 2D numpy array")
    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if K < 1:
        raise ValueError("K must be at least 1")
    if tol <= 0:
        raise ValueError("Tolerance must be positive")

    # Image resolution
    Hy, Hx = signal.shape
    X, Y = np.meshgrid(np.arange(1, Hx + 1) / Hx, np.arange(1, Hy + 1) / Hy)

    # Spectral Domain discretization
    fx = 1 / Hx
    fy = 1 / Hy
    freqs_1 = X - 0.5 - fx
    freqs_2 = Y - 0.5 - fy

    # Maximum number of iterations
    N = 100
  
    # Alpha might be individual for each mode
    Alpha = alpha * np.ones(K)

    # Construct f and f_hat
    f_hat = fftshift(fft2(signal))

    # Storage matrices for (Fourier) modes
    u_hat = np.zeros((Hy, Hx, K), dtype=complex)
    u_hat_old = np.copy(u_hat)
    sum_uk = 0

    # Storage matrices for (Fourier) Lagrange multiplier
    mu_hat = np.zeros((Hy, Hx), dtype=complex)

    # Initialize omega
    # omega = np.zeros((N, 2, K), dtype=complex)
    omega = np.zeros((N, 2, K))

    # Initialization of omega_k
    if init == 0:
        # Spread omegas radially
        maxK = K - 1 if DC else K
        for k in range(DC, maxK + DC):
            omega[0, 0, k] = 0.25 * np.cos(np.pi * (k - 1) / maxK)
            omega[0, 1, k] = 0.25 * np.sin(np.pi * (k - 1) / maxK)
    elif init == 1:
        # Random on half-plane
        omega[0, 0, :K] = np.random.rand(K) - 0.5

        omega[0, 1, :K] = np.random.rand(K) / 2

        # DC component (if expected)
        if DC == 1:
            omega[0, :, 0] = 0

    ## Main loop for iterative updates
    # Stopping criteria tolerances
    uDiff = tol + eps
    omegaDiff = tol + eps
    n = 0

    while (uDiff > tol or omegaDiff > tol) and n < N - 1:

        # First things first
        k = 0  # Python uses 0-based indexing

        # Compute the halfplane mask for the 2D "analytic signal"
        HilbertMask = (np.sign(freqs_1 * omega[n, 0, k] + freqs_2 * omega[n, 1, k]) + 1)

        # Update first mode accumulator
        sum_uk = u_hat[:, :, -1] + sum_uk - u_hat[:, :, k]

        # Update first mode's spectrum through Wiener filter (on half plane)
        u_hat[:, :, k] = ((f_hat - sum_uk - mu_hat / 2) * HilbertMask) / (1 + Alpha[k] * ((freqs_1 - omega[n, 0, k]) ** 2 + (freqs_2 - omega[n, 1, k]) ** 2))

        # Update first mode's central frequency as spectral center of gravity
        if not DC:
            omega[n + 1, 0, k] = np.sum(freqs_1 * np.abs(u_hat[:, :, k]) ** 2) / np.sum(np.abs(u_hat[:, :, k]) ** 2)
            omega[n + 1, 1, k] = np.sum(freqs_2 * np.abs(u_hat[:, :, k]) ** 2) / np.sum(np.abs(u_hat[:, :, k]) ** 2)
        
            # Keep omegas on the same halfplane
            if omega[n + 1, 1, k] < 0:
                omega[n + 1, :, k] = -omega[n + 1, :, k]

        # Recover full spectrum from analytic signal
        u_hat[:, :, k] = fftshift(fft2(np.real(ifft2(ifftshift(u_hat[:, :, k])))))

        # Work on other modes
        for k in range(1, K):
            # Recompute Hilbert mask
            HilbertMask = (np.sign(freqs_1 * omega[n, 0, k] + freqs_2 * omega[n, 1, k]) + 1)
        
            # Update accumulator
            sum_uk = u_hat[:, :, k - 1] + sum_uk - u_hat[:, :, k]
        
            # Update signal spectrum
            u_hat[:, :, k] = ((f_hat - sum_uk - mu_hat / 2) * HilbertMask) / (1 + Alpha[k] * ((freqs_1 - omega[n, 0, k]) ** 2 + (freqs_2 - omega[n, 1, k]) ** 2))

            # Update signal frequencies
            omega[n + 1, 0, k] = np.sum(freqs_1 * np.abs(u_hat[:, :, k]) ** 2) / np.sum(np.abs(u_hat[:, :, k]) ** 2)
            omega[n + 1, 1, k] = np.sum(freqs_2 * np.abs(u_hat[:, :, k]) ** 2) / np.sum(np.abs(u_hat[:, :, k]) ** 2)
        
            # Keep omegas on the same halfplane
            if omega[n + 1, 1, k] < 0:
                omega[n + 1, :, k] = -omega[n + 1, :, k]

            # Recover full spectrum from analytic signal
            u_hat[:, :, k] = fftshift(fft2(np.real(ifft2(ifftshift(u_hat[:, :, k])))))

        # Gradient ascent for augmented Lagrangian
        mu_hat = mu_hat + tau * (np.sum(u_hat, axis=2) - f_hat)

        # Increment iteration counter
        n += 1

        # Convergence?
        uDiff = eps
        omegaDiff = eps

        for k in range(K):
            omegaDiff += np.sum(np.abs(omega[n, :, k] - omega[n - 1, :, k]) ** 2)
            uDiff += np.sum(np.abs(u_hat[:, :, k] - u_hat_old[:, :, k]) ** 2) / (Hx * Hy)

        uDiff = np.abs(uDiff)
        u_hat_old = np.copy(u_hat)

        print(f"{n} time; uDiff: {uDiff} ; omegaDiff: {omegaDiff}")

    ## Signal Reconstruction

    # Inverse Fourier Transform to compute (spatial) modes
    u = np.zeros((Hy, Hx, K))
    for k in range(K):
        u[:, :, k] = np.real(ifft2(ifftshift(u_hat[:, :, k])))

    # Return final results
    return u, u_hat, omega

