"""
Avaliação de Desempenho - UECE - Trabalho Final (2026.2)
Questão Q1 - Plataforma IoT baseada em Nevoeiro (Fog-Based IoT Platform)

Modelo (Figura 1 / Tabela 1):
  - n = 10 nós Fog, filas M/M/1 (FCFS), taxa de serviço mu_i em [20, 30]
  - Servidor Cloud (n+1) e servidor A&F (n+2): filas M/M/infinito (Delay), taxa 200
  - Chegadas externas gamma_i (Poisson) em cada nó Fog i
  - Com prob. q_i = 0.05 a requisição vai direto ao Cloud; com 1-q_i entra no Fog i
  - Saindo do Fog i: com prob. p_i = 0.5 vai ao A&F, com 1-p_i vai ao Cloud
  - Saindo do Cloud, a requisição vai ao A&F; saindo do A&F, deixa o sistema

Métricas (curvas versus gamma total = soma dos gamma_i):
  L     = número médio de requisições no sistema (Fog + Cloud + A&F)
  L_fog = número médio de requisições nas filas Fog
  T     = tempo médio no sistema (Lei de Little: T = L / gamma)

Comparação dos métodos do LINE:
  - Solvers analíticos: MVA (exato), NC e MAM
  - Simulação: SSA (simulação estocástica de eventos discretos)
  (os solvers JMT/LDES exigem Java/binário nativo, indisponíveis aqui)

Execução:   python questao_q1.py            (completo)
            python questao_q1.py --rapido   (simulação mais curta)
"""

import sys
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # salva as figuras em arquivo (sem janela)
import matplotlib.pyplot as plt

from line_solver import (
    Network, Source, Queue, Delay, Sink, OpenClass, Exp, SchedStrategy,
    MVA, NC, MAM, SSA, GlobalConstants, VerboseLevel,
)

sys.stdout.reconfigure(encoding="utf-8")
GlobalConstants.set_verbose(VerboseLevel.SILENT)

# ----------------------------------------------------------------------------
# Parâmetros (Tabela 1)
# ----------------------------------------------------------------------------
N = 10                              # número de nós Fog
MU = np.linspace(20, 30, N)         # mu_i em [20, 30] (uniformemente espaçados)
MU_CLOUD = 200.0                    # mu_{n+1}
MU_AF = 200.0                       # mu_{n+2}
P = np.full(N, 0.5)                 # p_i: Fog -> A&F
Q = np.full(N, 0.05)                # q_i: externo -> Cloud direto
GAMMA_I = np.arange(1, 11)          # gamma_i = 1..10 (igual em todos os nós)

RAPIDO = "--rapido" in sys.argv
SSA_SAMPLES = 40_000 if RAPIDO else 200_000
SEED = 23000


# ----------------------------------------------------------------------------
# Construção do modelo no LINE
# ----------------------------------------------------------------------------
def constroi_modelo(gamma_i):
    """gamma_i: vetor (N,) com a taxa externa de cada nó Fog."""
    gamma_i = np.asarray(gamma_i, dtype=float)
    g_tot = gamma_i.sum()

    model = Network("FogIoT-Q1")
    src = Source(model, "Src")
    fog = [Queue(model, f"Fog{i + 1}", SchedStrategy.FCFS) for i in range(N)]
    cloud = Delay(model, "Cloud")   # M/M/infinito
    af = Delay(model, "AF")         # M/M/infinito
    sink = Sink(model, "Sink")

    c1 = OpenClass(model, "C1")
    # A superposição de fluxos Poisson independentes é Poisson; a Source emite
    # com taxa total e roteia para cada nó proporcionalmente a gamma_i.
    src.set_arrival(c1, Exp(g_tot))
    for i in range(N):
        fog[i].set_service(c1, Exp(MU[i]))
    cloud.set_service(c1, Exp(MU_CLOUD))
    af.set_service(c1, Exp(MU_AF))

    rt = model.init_routing_matrix()
    for i in range(N):
        rt.set(c1, c1, src, fog[i], (1 - Q[i]) * gamma_i[i] / g_tot)
        rt.set(c1, c1, fog[i], cloud, 1 - P[i])
        rt.set(c1, c1, fog[i], af, P[i])
    rt.set(c1, c1, src, cloud, float(np.sum(Q * gamma_i) / g_tot))
    rt.set(c1, c1, cloud, af, 1.0)
    rt.set(c1, c1, af, sink, 1.0)
    model.link(rt)
    return model


def metricas(solver, g_tot):
    """Extrai L, L_fog e T (Little) da tabela de médias do solver."""
    df = solver.avg_table().data           # DataFrame subjacente (IndexedTable)
    df = df[df["Station"] != "Src"]
    L = df["QLen"].sum()
    L_fog = df[df["Station"].str.startswith("Fog")]["QLen"].sum()
    return L, L_fog, L / g_tot


def analitico(gamma_i):
    """Solução de forma fechada (rede de Jackson) para validar os solvers."""
    lam_fog = (1 - Q) * gamma_i
    rho = lam_fog / MU
    L_fog = np.sum(rho / (1 - rho))                       # M/M/1
    lam_cloud = np.sum(Q * gamma_i + (1 - P) * lam_fog)
    lam_af = np.sum(P * lam_fog) + lam_cloud
    L = L_fog + lam_cloud / MU_CLOUD + lam_af / MU_AF     # M/M/inf: L = lambda/mu
    return L, L_fog, L / gamma_i.sum()


# ----------------------------------------------------------------------------
# Varredura em gamma
# ----------------------------------------------------------------------------
METODOS = {
    "MVA (exato)": lambda m: MVA(m),
    "NC": lambda m: NC(m),
    "MAM": lambda m: MAM(m),
    "Simulação (SSA)": lambda m: SSA(m, samples=SSA_SAMPLES, seed=SEED),
}


def main():
    linhas = []
    t0 = time.time()
    for g in GAMMA_I:
        gamma_i = np.full(N, float(g))
        g_tot = gamma_i.sum()
        model = constroi_modelo(gamma_i)
        reg = {"gamma_total": g_tot}
        La, Lfa, Ta = analitico(gamma_i)
        reg.update({"Analitico_L": La, "Analitico_Lfog": Lfa, "Analitico_T": Ta})
        for nome, fabrica in METODOS.items():
            L, Lf, T = metricas(fabrica(model), g_tot)
            reg.update({f"{nome}_L": L, f"{nome}_Lfog": Lf, f"{nome}_T": T})
        linhas.append(reg)
        print(f"gamma_total = {g_tot:5.0f} concluído ({time.time() - t0:6.1f}s)")

    res = pd.DataFrame(linhas)
    res.to_csv("resultados_q1.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\nResultados (também salvos em resultados_q1.csv):")
    print(res.round(5).to_string(index=False))

    # Validação dos solvers contra a forma fechada
    for nome in ("MVA (exato)", "NC", "MAM"):
        erro = np.abs(res[f"{nome}_L"] - res["Analitico_L"]).max()
        print(f"Erro máx. |L_{nome} - L_analítico| = {erro:.2e}")
    erro_sim = (np.abs(res["Simulação (SSA)_L"] - res["Analitico_L"])
                / res["Analitico_L"]).max()
    print(f"Erro relativo máx. da simulação em L = {100 * erro_sim:.2f}%")

    # ------------------------------------------------------------------------
    # Gráficos: L, L_fog e T versus gamma total
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
        ax.set_xlabel("Taxa de chegadas externas total γ (req/unid. tempo)")
        ax.set_ylabel(rotulo)
        ax.set_title(rotulo)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Q1 - Plataforma IoT baseada em Nevoeiro: Solvers do LINE vs. simulação")
    fig.tight_layout()
    fig.savefig("q1_curvas.png", dpi=150)
    print("Figura salva em q1_curvas.png")


if __name__ == "__main__":
    main()
