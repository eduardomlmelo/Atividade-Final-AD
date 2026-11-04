"""
Avaliação de Desempenho - UECE - Trabalho Final (2026.2)
Questão Q2 - Plataforma IoT baseada em Nevoeiro com 3 classes

Mesmo modelo de Q1 (n = 10 nós Fog M/M/1, Cloud e A&F M/M/infinito), porém com
3 classes de requisições. Acrescentam-se ao modelo original (classe C1):
  - 2 fluxos externos da classe C2 com mu = 40 e lambda = 5
  - 2 fluxos externos da classe C3 com mu = 50 e lambda = 4
As taxas mu_{n+1} e mu_{n+2} (Cloud e A&F) não mudam para C2 e C3 (= 200).

Hipóteses de modelagem (o enunciado não detalha onde os fluxos entram):
  - Cada "fluxo externo" é um fluxo Poisson que chega a um nó Fog distinto.
    C2 chega aos nós Fog 1 e 2; C3 chega aos nós Fog 3 e 4 (ver FLUXOS_EXTRA).
  - mu = 40 (C2) e mu = 50 (C3) são as taxas de serviço dessas classes nos nós
    Fog que as recebem; lambda = 5 e 4 são as taxas de chegada por fluxo.
  - Roteamento igual ao de C1 (q_i = 0.05, p_i = 0.5, Cloud -> A&F -> saída).
  - A varredura em gamma continua sendo a de C1 (gamma_i = 1..10 por nó); os
    fluxos de C2 e C3 permanecem fixos. O eixo x é a taxa externa TOTAL
    (C1 + C2 + C3).
  - Fog: FCFS com taxas de serviço dependentes da classe (não é produto-forma).
    MVA e MAM do LINE dão o resultado exato (conferido com Pollaczek-Khinchine
    M/G/1 abaixo); o NC é aproximado neste caso.

Métricas: L, L_fog e T (T = L / gamma total, Lei de Little), além de L e T por
classe (T_c = L_c / gamma_c).

Execução:   python questao_q2.py            (completo)
            python questao_q2.py --rapido   (simulação mais curta)
"""

import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from line_solver import (
    Network, Source, Queue, Delay, Sink, OpenClass, Exp, SchedStrategy,
    MVA, NC, MAM, SSA, GlobalConstants, VerboseLevel,
)

sys.stdout.reconfigure(encoding="utf-8")
GlobalConstants.set_verbose(VerboseLevel.SILENT)

# ----------------------------------------------------------------------------
# Parâmetros
# ----------------------------------------------------------------------------
N = 10
MU_C1 = np.linspace(20, 30, N)      # mu_i da classe C1 em cada nó Fog
MU_CLOUD = 200.0
MU_AF = 200.0
P = np.full(N, 0.5)
Q = np.full(N, 0.05)
GAMMA_I = np.arange(1, 11)          # gamma_i de C1 = 1..10

# (classe, lambda por fluxo, mu da classe, nós Fog (0-indexado) que recebem)
FLUXOS_EXTRA = [
    ("C2", 5.0, 40.0, [0, 1]),
    ("C3", 4.0, 50.0, [2, 3]),
]
CLASSES = ["C1", "C2", "C3"]

RAPIDO = "--rapido" in sys.argv
SSA_SAMPLES = 40_000 if RAPIDO else 200_000
SEED = 23000


def taxas_por_classe(g):
    """gam[k][i] = taxa externa da classe k no nó Fog i; mu[k][i] = serviço."""
    gam = np.zeros((3, N))
    mu = np.zeros((3, N))
    gam[0, :] = g
    mu[0, :] = MU_C1
    for k, (_, lam, muk, nos) in enumerate(FLUXOS_EXTRA, start=1):
        gam[k, nos] = lam
        mu[k, :] = muk              # só é usado onde há fluxo da classe
    return gam, mu


# ----------------------------------------------------------------------------
# Construção do modelo no LINE
# ----------------------------------------------------------------------------
def constroi_modelo(gam, mu):
    model = Network("FogIoT-Q2")
    src = Source(model, "Src")
    fog = [Queue(model, f"Fog{i + 1}", SchedStrategy.FCFS) for i in range(N)]
    cloud = Delay(model, "Cloud")
    af = Delay(model, "AF")
    sink = Sink(model, "Sink")
    cls = [OpenClass(model, nome) for nome in CLASSES]

    for k, c in enumerate(cls):
        src.set_arrival(c, Exp(gam[k].sum()))
        for i in range(N):
            fog[i].set_service(c, Exp(mu[k, i]))
        cloud.set_service(c, Exp(MU_CLOUD))
        af.set_service(c, Exp(MU_AF))

    rt = model.init_routing_matrix()
    for k, c in enumerate(cls):
        tot = gam[k].sum()
        for i in range(N):
            if gam[k, i] > 0:
                rt.set(c, c, src, fog[i], (1 - Q[i]) * gam[k, i] / tot)
            rt.set(c, c, fog[i], cloud, 1 - P[i])
            rt.set(c, c, fog[i], af, P[i])
        rt.set(c, c, src, cloud, float(np.sum(Q * gam[k]) / tot))
        rt.set(c, c, cloud, af, 1.0)
        rt.set(c, c, af, sink, 1.0)
    model.link(rt)
    return model


def metricas(solver, gam):
    """Retorna dict com L, L_fog, T (totais) e L_c, T_c por classe."""
    df = solver.avg_table().data
    df = df[df["Station"] != "Src"]
    g_cls = gam.sum(axis=1)
    g_tot = g_cls.sum()
    L = df["QLen"].sum()
    L_fog = df[df["Station"].str.startswith("Fog")]["QLen"].sum()
    out = {"L": L, "Lfog": L_fog, "T": L / g_tot}
    for k, nome in enumerate(CLASSES):
        Lc = df[df["JobClass"] == nome]["QLen"].sum()
        out[f"L_{nome}"] = Lc
        out[f"T_{nome}"] = Lc / g_cls[k]
    return out


def analitico(gam, mu):
    """Forma fechada para conferência.

    Fog i: M/G/1 FCFS com serviço hiperexponencial (mistura das classes) ->
    L = rho + lam^2 E[S^2] / (2 (1 - rho)).  Cloud/A&F: M/M/inf (L = lam/mu).
    """
    L_fog = 0.0
    for i in range(N):
        lam_k = (1 - Q[i]) * gam[:, i]
        lam = lam_k.sum()
        if lam == 0:
            continue
        ES = np.sum(lam_k / mu[:, i]) / lam
        ES2 = np.sum(lam_k * 2.0 / mu[:, i] ** 2) / lam
        rho = lam * ES
        L_fog += rho + lam ** 2 * ES2 / (2 * (1 - rho))
    lam_fog = (1 - Q) * gam.sum(axis=0)
    lam_cloud = np.sum(Q * gam.sum(axis=0) + (1 - P) * lam_fog)
    lam_af = np.sum(P * lam_fog) + lam_cloud
    L = L_fog + lam_cloud / MU_CLOUD + lam_af / MU_AF
    return L, L_fog, L / gam.sum()


METODOS = {
    "MVA (exato)": lambda m: MVA(m),
    "NC": lambda m: NC(m),
    "MAM": lambda m: MAM(m),
    "Simulação (SSA)": lambda m: SSA(m, samples=SSA_SAMPLES, seed=SEED),
}


def main():
    # Checagem de estabilidade dos nós Fog no pior caso (gamma_i = 10)
    gam, mu = taxas_por_classe(float(GAMMA_I.max()))
    rho_max = (((1 - Q) * gam) / mu).sum(axis=0).max()
    print(f"Utilização máxima de um nó Fog no pior caso: {rho_max:.3f} (< 1, estável)")

    linhas = []
    t0 = time.time()
    for g in GAMMA_I:
        gam, mu = taxas_por_classe(float(g))
        g_tot = gam.sum()
        model = constroi_modelo(gam, mu)
        reg = {"gamma_total": g_tot, "gamma_C1_por_no": float(g)}
        La, Lfa, Ta = analitico(gam, mu)
        reg.update({"Analitico_L": La, "Analitico_Lfog": Lfa, "Analitico_T": Ta})
        for nome, fabrica in METODOS.items():
            for chave, val in metricas(fabrica(model), gam).items():
                reg[f"{nome}_{chave}"] = val
        linhas.append(reg)
        print(f"gamma_total = {g_tot:5.0f} concluído ({time.time() - t0:6.1f}s)")

    res = pd.DataFrame(linhas)
    res.to_csv("resultados_q2.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    colunas = ["gamma_total", "Analitico_L"] + [
        f"{m}_{s}" for m in METODOS for s in ("L", "Lfog", "T")]
    print("\nResultados (todas as colunas em resultados_q2.csv):")
    print(res[colunas].round(5).to_string(index=False))

    for nome in ("MVA (exato)", "NC", "MAM"):
        erro = (np.abs(res[f"{nome}_L"] - res["Analitico_L"])
                / res["Analitico_L"]).max()
        print(f"Erro relativo máx. de L ({nome}) vs. forma fechada = {100 * erro:.3e}%")
    erro_sim = (np.abs(res["Simulação (SSA)_L"] - res["Analitico_L"])
                / res["Analitico_L"]).max()
    print(f"Erro relativo máx. da simulação em L = {100 * erro_sim:.2f}%")

    # ------------------------------------------------------------------------
    # Figura 1: L, L_fog e T (totais) versus gamma total
    # ------------------------------------------------------------------------
    estilos = {
        "MVA (exato)": dict(color="tab:blue", marker="o", ls="-"),
        "NC": dict(color="tab:green", marker="s", ls="--"),
        "MAM": dict(color="tab:orange", marker="^", ls=":"),
        "Simulação (SSA)": dict(color="tab:red", marker="x", ls="-."),
    }
    painel = [("L", "L (número no sistema)"),
              ("Lfog", "L_fog (número nas filas Fog)"),
              ("T", "T (tempo no sistema)")]
    fig, eixos = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (suf, rotulo) in zip(eixos, painel):
        for nome, est in estilos.items():
            ax.plot(res["gamma_total"], res[f"{nome}_{suf}"], label=nome,
                    markersize=6, **est)
        ax.set_xlabel("Taxa de chegadas externas total γ (C1+C2+C3)")
        ax.set_ylabel(rotulo)
        ax.set_title(rotulo)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Q2 - Modelo com 3 classes: Solvers do LINE vs. simulação")
    fig.tight_layout()
    fig.savefig("q2_curvas.png", dpi=150)

    # ------------------------------------------------------------------------
    # Figura 2: L_c e T_c por classe (MVA exato vs. simulação)
    # ------------------------------------------------------------------------
    cores = {"C1": "tab:blue", "C2": "tab:orange", "C3": "tab:green"}
    fig2, eixos2 = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (pref, rotulo) in zip(eixos2, [("L", "L por classe"), ("T", "T por classe")]):
        for nome in CLASSES:
            ax.plot(res["gamma_total"], res[f"MVA (exato)_{pref}_{nome}"],
                    color=cores[nome], marker="o", ls="-", label=f"{nome} - MVA")
            ax.plot(res["gamma_total"], res[f"Simulação (SSA)_{pref}_{nome}"],
                    color=cores[nome], marker="x", ls="-.", label=f"{nome} - Simulação")
        ax.set_xlabel("Taxa de chegadas externas total γ (C1+C2+C3)")
        ax.set_ylabel(rotulo)
        ax.set_title(rotulo)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig2.suptitle("Q2 - Métricas por classe")
    fig2.tight_layout()
    fig2.savefig("q2_curvas_por_classe.png", dpi=150)
    print("Figuras salvas em q2_curvas.png e q2_curvas_por_classe.png")


if __name__ == "__main__":
    main()
