# Quantum_dyson_smatrix_qed_regressor 



import random
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_QUBITS = 6
STATE_DIM = 2 ** NUM_QUBITS
MAX_NUM = 39
NUM_POSITIONS = 7
DYSON_ORDER_STEPS = 4
COUPLING_GRID = (0.20, 0.45)
OMEGA_GRID = (0.35, 0.80)


def load_draws(file_path: Path) -> np.ndarray:
    rows = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            clean = line.strip()
            if not clean:
                continue
            parts = [p.strip() for p in clean.split(",") if p.strip()]
            nums = [int(x) for x in parts[:NUM_POSITIONS]]
            if len(nums) != NUM_POSITIONS:
                raise ValueError(f"Neispravan broj elemenata u redu: {clean}")
            if len(set(nums)) != NUM_POSITIONS or any(n < 1 or n > MAX_NUM for n in nums):
                raise ValueError(f"Neispravni brojevi u redu: {clean}")
            rows.append(sorted(nums))
    data = np.array(rows, dtype=np.int64)
    if len(data) < DYSON_ORDER_STEPS + 35:
        raise ValueError(f"Nedovoljno istorijskih kola u fajlu: {file_path}")
    return data


def get_position_mask() -> np.ndarray:
    num_grid = np.arange(1, MAX_NUM + 1, dtype=np.int64)[None, :]
    pos_idx = np.arange(NUM_POSITIONS, dtype=np.int64)[:, None]
    min_allowed = pos_idx + 1
    max_allowed = MAX_NUM - (NUM_POSITIONS - 1 - pos_idx)
    return (num_grid >= min_allowed) & (num_grid <= max_allowed)


def build_free_hamiltonian_spectrum(draws_train: np.ndarray) -> np.ndarray:
    counts = np.ones(MAX_NUM, dtype=np.float64)
    for row in draws_train:
        for val in row:
            counts[int(val) - 1] += 1.0
    probs = counts / np.sum(counts)
    energies = -np.log(probs)
    energies = (energies - float(np.mean(energies))) / max(float(np.std(energies)), 1e-8)
    return energies


def build_vertex_operators(draws: np.ndarray) -> np.ndarray:
    n_total = len(draws)
    v_ops = np.zeros((n_total, MAX_NUM, MAX_NUM), dtype=np.complex128)
    inv_norm = 1.0 / float(NUM_POSITIONS)

    for t in range(n_total):
        draw_zero = [int(x) - 1 for x in draws.__getitem__(t)]
        mat = np.zeros((MAX_NUM, MAX_NUM), dtype=np.complex128)
        for pos, k in enumerate(draw_zero):
            mat[k, k] = 0.5 * (float(pos + 1) / float(NUM_POSITIONS))
        for idx_a, idx_b in combinations(range(NUM_POSITIONS), 2):
            u = int(draw_zero.__getitem__(idx_a))
            v = int(draw_zero.__getitem__(idx_b))
            phase = np.exp(1j * float(idx_b - idx_a) * np.pi / float(NUM_POSITIONS))
            mat[u, v] = inv_norm * phase
            mat[v, u] = inv_norm * np.conj(phase)
        v_ops[t] = mat

    return v_ops


def polar_unitarize_subspace(sub_op: np.ndarray) -> np.ndarray:
    u_svd, _, vh_svd = np.linalg.svd(sub_op)
    return u_svd @ vh_svd


def compute_dyson_smatrix_scattering(
    v_ops: np.ndarray,
    draws: np.ndarray,
    h0_diag: np.ndarray,
    coupling_g: float,
    omega_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_total = len(draws)
    n_samples = n_total - DYSON_ORDER_STEPS

    time_steps = np.arange(DYSON_ORDER_STEPS, dtype=np.float64) - float(DYSON_ORDER_STEPS - 1)
    u0_phases = np.exp(1j * omega_scale * time_steps[:, None] * h0_diag[None, :])
    u0_conj = np.conj(u0_phases)

    eye_sub = np.eye(MAX_NUM, dtype=np.complex128)
    in_phases = np.exp(1j * np.arange(NUM_POSITIONS, dtype=np.float64) * np.pi / float(NUM_POSITIONS))
    in_norm = float(np.sqrt(NUM_POSITIONS))

    scatter_dens = np.empty((n_samples, MAX_NUM), dtype=np.float64)
    y_targets = np.empty((n_samples, NUM_POSITIONS), dtype=np.int64)

    for idx in range(n_samples):
        t = idx + DYSON_ORDER_STEPS
        v_int_seq = np.empty((DYSON_ORDER_STEPS, MAX_NUM, MAX_NUM), dtype=np.complex128)
        for m in range(DYSON_ORDER_STEPS):
            v_bare = v_ops.__getitem__(t - DYSON_ORDER_STEPS + m)
            p_left = u0_phases.__getitem__(m)[:, None]
            p_right = u0_conj.__getitem__(m)[None, :]
            v_int_seq[m] = p_left * v_bare * p_right

        first_order = np.sum(v_int_seq, axis=0)
        second_order = np.zeros((MAX_NUM, MAX_NUM), dtype=np.complex128)
        for m1 in range(DYSON_ORDER_STEPS - 1):
            v_earlier = v_int_seq.__getitem__(m1)
            for m2 in range(m1 + 1, DYSON_ORDER_STEPS):
                v_later = v_int_seq.__getitem__(m2)
                second_order += v_later @ v_earlier

        s_series = eye_sub - 1j * coupling_g * first_order - (coupling_g ** 2) * second_order
        s_unitary_sub = polar_unitarize_subspace(s_series)

        psi_in = np.zeros(MAX_NUM, dtype=np.complex128)
        last_draw = draws.__getitem__(t - 1) - 1
        psi_in[last_draw] = in_phases / in_norm

        psi_out = s_unitary_sub @ psi_in
        scatter_dens[idx] = np.abs(psi_out) ** 2
        y_targets[idx] = draws.__getitem__(t)

    v_int_latest = np.empty((DYSON_ORDER_STEPS, MAX_NUM, MAX_NUM), dtype=np.complex128)
    for m in range(DYSON_ORDER_STEPS):
        v_bare = v_ops.__getitem__(n_total - DYSON_ORDER_STEPS + m)
        p_left = u0_phases.__getitem__(m)[:, None]
        p_right = u0_conj.__getitem__(m)[None, :]
        v_int_latest[m] = p_left * v_bare * p_right

    first_lat = np.sum(v_int_latest, axis=0)
    second_lat = np.zeros((MAX_NUM, MAX_NUM), dtype=np.complex128)
    for m1 in range(DYSON_ORDER_STEPS - 1):
        v_earlier = v_int_latest.__getitem__(m1)
        for m2 in range(m1 + 1, DYSON_ORDER_STEPS):
            v_later = v_int_latest.__getitem__(m2)
            second_lat += v_later @ v_earlier

    s_latest_series = eye_sub - 1j * coupling_g * first_lat - (coupling_g ** 2) * second_lat
    s_latest_sub = polar_unitarize_subspace(s_latest_series)

    s_full_unitary = np.eye(STATE_DIM, dtype=np.complex128)
    s_full_unitary[:MAX_NUM, :MAX_NUM] = s_latest_sub

    psi_in_latest = np.zeros(STATE_DIM, dtype=np.complex128)
    latest_draw = draws.__getitem__(n_total - 1) - 1
    psi_in_latest[latest_draw] = in_phases / in_norm

    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(UnitaryGate(s_full_unitary, label="DysonSMat"), list(range(NUM_QUBITS)))
    psi_out_qiskit = Statevector(psi_in_latest).evolve(qc).data
    latest_scatter_dens = (np.abs(psi_out_qiskit[:MAX_NUM]) ** 2)[None, :]

    return scatter_dens, y_targets, latest_scatter_dens


def scattering_to_position_probs(
    scatter_dens: np.ndarray,
    channel_gain: np.ndarray,
    pos_bias: np.ndarray,
    pos_mask: np.ndarray,
) -> np.ndarray:
    log_s = np.log(np.maximum(scatter_dens, 1e-12))
    logits = channel_gain[None, :, None] * log_s[:, None, :] + pos_bias[None, :, :]
    logits_masked = np.where(pos_mask[None, :, :], logits, -1e9)
    logits_shifted = logits_masked - np.max(logits_masked, axis=2, keepdims=True)
    exp_l = np.where(pos_mask[None, :, :], np.exp(np.clip(logits_shifted, -30.0, 30.0)), 0.0)
    denom = np.sum(exp_l, axis=2, keepdims=True)
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]
    return np.where(denom > 1e-15, exp_l / np.maximum(denom, 1e-15), uniform_fb)





def fit_qed_scattering_detector(
    dens_train: np.ndarray,
    y_train: np.ndarray,
    pos_mask: np.ndarray,
    maxiter: int = 55,
) -> dict:
    n_samples = len(y_train)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_train - 1

    one_hot = np.zeros((n_samples, NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    one_hot[sample_idx, pos_idx, target_idx] = 1.0

    log_s = np.log(np.maximum(dens_train, 1e-12))
    uniform_fb = pos_mask[None, :, :].astype(np.float64) / np.sum(pos_mask, axis=1, keepdims=True)[None, :, :]

    init_gain = np.ones(NUM_POSITIONS, dtype=np.float64)
    init_bias = np.zeros((NUM_POSITIONS, MAX_NUM), dtype=np.float64)
    init_params = np.concatenate([init_gain, init_bias.reshape(-1)])

    def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        gain = params[:NUM_POSITIONS]
        bias = params[NUM_POSITIONS:].reshape(NUM_POSITIONS, MAX_NUM)

        logits = gain[None, :, None] * log_s[:, None, :] + bias[None, :, :]
        logits_masked = np.where(pos_mask[None, :, :], logits, -1e9)
        logits_shifted = logits_masked - np.max(logits_masked, axis=2, keepdims=True)
        exp_l = np.where(pos_mask[None, :, :], np.exp(np.clip(logits_shifted, -30.0, 30.0)), 0.0)
        denom = np.sum(exp_l, axis=2, keepdims=True)
        probs = np.where(denom > 1e-15, exp_l / np.maximum(denom, 1e-15), uniform_fb)

        chosen = probs[sample_idx, pos_idx, target_idx]
        nll = -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))
        reg = 1e-3 * float(np.mean(bias ** 2)) + 1e-4 * float(np.mean((gain - 1.0) ** 2))
        total_loss = nll + reg

        d_logits = np.where(pos_mask[None, :, :], probs - one_hot, 0.0) / float(n_samples)
        grad_gain = np.sum(d_logits * log_s[:, None, :], axis=(0, 2)) + (2e-4 / float(NUM_POSITIONS)) * (gain - 1.0)
        grad_bias = np.sum(d_logits, axis=0) + (2e-3 / float(NUM_POSITIONS * MAX_NUM)) * bias

        return total_loss, np.concatenate([grad_gain, grad_bias.reshape(-1)])

    res = minimize(
        loss_and_grad,
        init_params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": maxiter, "ftol": 1e-7, "gtol": 1e-5},
    )

    opt = res.x
    return {
        "channel_gain": opt[:NUM_POSITIONS],
        "pos_bias": opt[NUM_POSITIONS:].reshape(NUM_POSITIONS, MAX_NUM),
        "pos_mask": pos_mask,
    }


def evaluate_qed_nll(model: dict, dens_eval: np.ndarray, y_eval: np.ndarray) -> float:
    probs = scattering_to_position_probs(
        dens_eval,
        model.__getitem__("channel_gain"),
        model.__getitem__("pos_bias"),
        model.__getitem__("pos_mask"),
    )
    n_samples = len(y_eval)
    sample_idx = np.arange(n_samples)[:, None]
    pos_idx = np.arange(NUM_POSITIONS)[None, :]
    target_idx = y_eval - 1
    chosen = probs[sample_idx, pos_idx, target_idx]
    return -float(np.mean(np.sum(np.log(np.maximum(chosen, 1e-12)), axis=1)))


def decode_smatrix_scattering_channels(pos_probs: np.ndarray, pos_mask: np.ndarray) -> list[int]:
    neg_inf = -1e18
    dp = np.full((NUM_POSITIONS, MAX_NUM), neg_inf, dtype=np.float64)
    parent = np.full((NUM_POSITIONS, MAX_NUM), -1, dtype=np.int64)

    first_mask = pos_mask.__getitem__(0)
    first_p = pos_probs.__getitem__(0)
    for k in range(MAX_NUM):
        if bool(first_mask.__getitem__(k)):
            dp[0, k] = np.log(max(float(first_p.__getitem__(k)), 1e-12))

    for pos in range(1, NUM_POSITIONS):
        curr_mask = pos_mask.__getitem__(pos)
        curr_p = pos_probs.__getitem__(pos)
        prev_dp = dp.__getitem__(pos - 1)
        for k in range(MAX_NUM):
            if not bool(curr_mask.__getitem__(k)):
                continue
            best_prev = neg_inf
            best_idx = -1
            for prev_k in range(k):
                val = float(prev_dp.__getitem__(prev_k))
                if val > best_prev:
                    best_prev = val
                    best_idx = prev_k
            if best_idx >= 0 and best_prev > neg_inf * 0.5:
                dp[pos, k] = best_prev + np.log(max(float(curr_p.__getitem__(k)), 1e-12))
                parent[pos, k] = best_idx

    last_dp = dp.__getitem__(NUM_POSITIONS - 1)
    best_last = int(np.argmax(last_dp))
    if float(last_dp.__getitem__(best_last)) <= neg_inf * 0.5:
        raise RuntimeError("Neuspesno dekodiranje kanala S-matrice.")

    seq_rev: list[int] = []
    curr_k = best_last
    for pos in range(NUM_POSITIONS - 1, -1, -1):
        seq_rev.append(curr_k + 1)
        curr_k = int(parent[pos, curr_k])

    seq = list(reversed(seq_rev))
    if len(set(seq)) != NUM_POSITIONS:
        raise RuntimeError("Dekodirana kombinacija nema 7 razlicitih brojeva.")
    return seq


def main() -> None:
    print("Metoda: Dyson Time-Ordered QED S-Matrix Scattering Regressor (1x6 qubita)")
    print("Posebnost: Interakciona slika V_I(t) + Dysonov red 2. stepena T*exp(-i int V_I) na Qiskit simulatoru")

    pos_mask = get_position_mask()

    for file_path in FILES:
        if not file_path.exists():
            print(f"Fajl nije pronadjen: {file_path}")
            continue

        draws = load_draws(file_path)
        v_ops = build_vertex_operators(draws)
        n_samples = len(draws) - DYSON_ORDER_STEPS

        train_end = int(0.75 * n_samples)
        val_end = int(0.88 * n_samples)

        h0_tr = build_free_hamiltonian_spectrum(draws[: train_end + DYSON_ORDER_STEPS])

        best_g = float(COUPLING_GRID.__getitem__(0))
        best_omega = float(OMEGA_GRID.__getitem__(0))
        best_val_nll = float("inf")
        cached_scatter: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

        for g_val in COUPLING_GRID:
            for omega_val in OMEGA_GRID:
                dens_all, y_all, _ = compute_dyson_smatrix_scattering(
                    v_ops,
                    draws,
                    h0_tr,
                    coupling_g=g_val,
                    omega_scale=omega_val,
                )
                cached_scatter[(g_val, omega_val)] = (dens_all, y_all)

                model = fit_qed_scattering_detector(dens_all[:train_end], y_all[:train_end], pos_mask, maxiter=55)
                val_nll = evaluate_qed_nll(model, dens_all[train_end:val_end], y_all[train_end:val_end])
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_g = g_val
                    best_omega = omega_val

        dens_best, y_best = cached_scatter.__getitem__((best_g, best_omega))
        eval_model = fit_qed_scattering_detector(dens_best[:train_end], y_best[:train_end], pos_mask, maxiter=55)
        test_nll = evaluate_qed_nll(eval_model, dens_best[val_end:], y_best[val_end:])

        h0_full = build_free_hamiltonian_spectrum(draws)
        dens_full, y_full, latest_dens = compute_dyson_smatrix_scattering(
            v_ops,
            draws,
            h0_full,
            coupling_g=best_g,
            omega_scale=best_omega,
        )
        final_model = fit_qed_scattering_detector(dens_full, y_full, pos_mask, maxiter=55)

        next_probs = scattering_to_position_probs(
            latest_dens,
            final_model.__getitem__("channel_gain"),
            final_model.__getitem__("pos_bias"),
            pos_mask,
        ).__getitem__(0)
        next_combo = decode_smatrix_scattering_channels(next_probs, pos_mask)

        print(f"CSV: {file_path.name}")
        print(f"Validacija NLL (g={best_g}, omega={best_omega}): {best_val_nll:.4f}")
        print(f"Test NLL: {test_nll:.4f}")
        print(f"NEXT: {next_combo}")


if __name__ == "__main__":
    main()



"""
Metoda: Dyson Time-Ordered QED S-Matrix Scattering Regressor (1x6 qubita)
Posebnost: Interakciona slika V_I(t) + Dysonov red 2. stepena T*exp(-i int V_I) na Qiskit simulatoru

CSV: loto7_4718_k90.csv
Validacija NLL (g=0.45, omega=0.8): 20.6582
Test NLL: 20.5892
NEXT: [1, x, 13, y, 29, z, 38]

CSV: loto7_4718_k90_loto_2981.csv
Validacija NLL (g=0.2, omega=0.8): 20.6901
Test NLL: 20.7206
NEXT: [1, x, 15, y, 24, z, 39]

CSV: loto7_4718_k90_loto_plus_1737.csv
Validacija NLL (g=0.45, omega=0.35): 20.3405
Test NLL: 20.5332
NEXT: [1, x, 11, y, 27, z, 39]
"""



"""
Freeman John Dyson
unify quantum electrodynamics
Sin-Itiro Tomonaga, Julian Schwinger, Richard Feynman


Freeman John Dyson — Ujedinjenje kvantne elektrodinamike (Dyson-ov vremenski uređeni S-matrični red \(\mathcal{T}\exp\left(-i \int \hat{H}_I(t)\,dt\right)\) i spajanje Schwinger-Tomonaga operatora sa Feynman-ovim propagatorima):

• Oblast Dyson-ovog S-matričnog razvoja i interakcione slike (Interaction Picture, Dyson-ov vremenski uređeni eksponent \(\hat{S} = \mathcal{T}\exp(-i \int \hat{V}_I(t)\,dt) = \hat{I} - i\int \hat{V}_I(t_1)\,dt_1 + (-i)^2 \iint_{t_1 > t_2} \hat{V}_I(t_1)\hat{V}_I(t_2)\,dt_1 dt_2 + \dots\))

• Osnovna ideja (Freeman Dyson, 1949):
	• Dyson je pokazao da su Schwinger-Tomonaga varijacioni formalizam i Feynman-ovi dijagrami po putanjama matematički ekvivalentni kada se sistem posmatra u interakcionoj slici (Dirac-Dyson picture):
    \(\^{V}_{I}(t)=e^{i\^{H}_{0}t}\^{V}(t)e^{-i\^{H}_{0}t}\)
	• Evolucija sistema kroz uzastopne vremenske korake \(t_1 < t_2 < \dots < t_M\) opisana je Dyson-ovim vremenski uređenim operatorom rasejanja (S-matricom) do 2. reda (jednopetljasti / two-vertex exchange):
    \(\^{S}^{(2)}=\^{I}-i\sum _{m=1}^{M}\^{V}_{I}(t_{m})-\sum _{1\le m_{1}<m_{2}\le M}\^{V}_{I}(t_{m_{2}})\^{V}_{I}(t_{m_{1}})\)
	• Prvi član (\(-i \sum \hat{V}_I(t_m)\)) predstavlja direktno rasejanje prvog reda (tree-level), dok vremenski uređeni proizvod \(\hat{V}_I(t_{m_2})\hat{V}_I(t_{m_1})\) (\(t_{m_2} > t_{m_1}\)) hvata nekomutativne dvokoračne korelacije u vremenu (kada redosled pojavljivanja brojeva u kolima \(t_{m_{1}}\) pa \(t_{m_{2}}\) menja amplitudu rasejanja!).

	1. Slobodni Hamiltonijan \(\^{H}_{0}\) i interakcioni potencijal \(\hat{V}(t)\) u prostoru 39 brojeva: Iz hronološke istorije (isključivo na trening skupu bez curenja podataka) definiše se slobodni diagonalni spektar \(\^{H}_{0}\) i nekomutativni interakcioni operator \(\hat{V}(t)\) svakog kola.
	2. Qiskit Dyson S-Matrix Evolution Circuit (UnitaryGate sa polarno-SVD re-unitarizacijom na 1x6 qubita): Dyson-ov vremenski uređeni operator \(\^{S}^{(2)}\) za hronološki prozor od M kola (gde redosled \(t_1 < t_2 < \dots < t_M\) strogo poštuje \(\mathcal{T}\)-uređenje!) projektuje se na najbližu unitarnu S-matricu \(\hat{S}_{\text{QED}} \in U(64)\) i izvršava na 6-qubitnom Qiskit simulatoru.
	3. S-matrična prelazna verovatnoća rasejanja \(\lvert \langle k \mid \hat{S}_{\text{QED}} \mid \psi_{\text{in}} \rangle \rvert^2\) i dekoracija kombinacije: Ulazno stanje \(\lvert \psi_{\text{in}} \rangle\) poslednjeg kola rasejava se kroz \(\^{S}_{\text{QED}}\) u izlazni spektar rasejanja na 39 kanala, a NEXT kombinacija od 7 različitih brojeva dobija se S-Matrix Scattering Channel Selection algoritmom uz očuvanje strogog poretka \(1 \le x_1 < \dots < x_7 \le 39\).

Dyson-ov vremenski uređeni proizvod \(\mathcal{T}\{\hat{V}_I(t_2)\hat{V}_I(t_1)\}\) eksplicitno razlikuje uzlazni od silaznog poretka prelaza kroz nekomutativnost operatora u interakcionoj slici.
"""
