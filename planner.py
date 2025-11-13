"""Planner de equilíbrio vida-trabalho utilizando Programação Dinâmica.

Este script cria um conjunto de tarefas, organiza-as com Merge Sort recursivo
e aplica o problema da Mochila 0/1 para maximizar o impacto de bem-estar
dentro de um limite diário de tempo.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


console = Console()


TaskRecord = Dict[str, object]


def gerar_tarefas_base() -> List[TaskRecord]:
    """Retorna a lista fixa de tarefas disponíveis."""
    return [
        {"task_id": 1, "name": "Meditação guiada", "duration": 20, "priority": 5, "type": "pessoal", "impact": 3},
        {"task_id": 2, "name": "Leitura técnica", "duration": 45, "priority": 4, "type": "profissional", "impact": 2},
        {"task_id": 3, "name": "Revisão de metas semanais", "duration": 30, "priority": 5, "type": "profissional", "impact": 3},
        {"task_id": 4, "name": "Treino funcional", "duration": 60, "priority": 4, "type": "pessoal", "impact": 3},
        {"task_id": 5, "name": "Reunião estratégica", "duration": 90, "priority": 5, "type": "profissional", "impact": 2},
        {"task_id": 6, "name": "Escrever diário", "duration": 25, "priority": 3, "type": "pessoal", "impact": 2},
        {"task_id": 7, "name": "Mentoria rápida", "duration": 35, "priority": 3, "type": "profissional", "impact": 2},
        {"task_id": 8, "name": "Cozinhar almoço saudável", "duration": 50, "priority": 4, "type": "pessoal", "impact": 3},
        {"task_id": 9, "name": "Alongamento ativo", "duration": 15, "priority": 2, "type": "pessoal", "impact": 1},
        {"task_id": 10, "name": "Código foco profundo", "duration": 120, "priority": 5, "type": "profissional", "impact": 3},
        {"task_id": 11, "name": "Responder e-mails críticos", "duration": 40, "priority": 3, "type": "profissional", "impact": 2},
        {"task_id": 12, "name": "Brincar com filhos", "duration": 45, "priority": 4, "type": "pessoal", "impact": 3},
        {"task_id": 13, "name": "Networking virtual", "duration": 30, "priority": 2, "type": "profissional", "impact": 1},
        {"task_id": 14, "name": "Curso online curto", "duration": 70, "priority": 4, "type": "profissional", "impact": 2},
        {"task_id": 15, "name": "Passeio ao ar livre", "duration": 60, "priority": 3, "type": "pessoal", "impact": 2},
        {"task_id": 16, "name": "Planejamento financeiro", "duration": 35, "priority": 4, "type": "pessoal", "impact": 3},
        {"task_id": 17, "name": "Design review", "duration": 55, "priority": 3, "type": "profissional", "impact": 2},
        {"task_id": 18, "name": "Podcast inspirador", "duration": 30, "priority": 2, "type": "pessoal", "impact": 1},
        {"task_id": 19, "name": "Chamada com a liderança", "duration": 40, "priority": 4, "type": "profissional", "impact": 2},
        {"task_id": 20, "name": "Sessão de foco criativo", "duration": 80, "priority": 5, "type": "profissional", "impact": 3},
        {"task_id": 21, "name": "Organizar espaço de trabalho", "duration": 25, "priority": 3, "type": "pessoal", "impact": 2},
        {"task_id": 22, "name": "Almoço com amigo", "duration": 60, "priority": 2, "type": "pessoal", "impact": 2},
    ]


SortPlan = Sequence[Tuple[str, bool]]


def parse_cli_args() -> argparse.Namespace:
    """Captura parâmetros opcionais informados pelo usuário."""
    parser = argparse.ArgumentParser(
        description="Planejador de equilíbrio entre vida pessoal e profissional usando Programação Dinâmica."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=480,
        help="Limite diário de tempo em minutos (padrão: 480).",
    )
    parser.add_argument(
        "--sort",
        choices=["impacto", "prioridade", "hibrido"],
        default="hibrido",
        help="Estratégia de ordenação antes da mochila (padrão: hibrido).",
    )
    parser.add_argument(
        "--html-report",
        type=str,
        default=None,
        help="Caminho opcional para exportar um relatório HTML estilizado.",
    )
    return parser.parse_args()


def obter_plano_ordenacao(modo: str) -> SortPlan:
    """Define o plano de ordenação com base na preferência do usuário."""
    if modo == "impacto":
        return (("impact", True), ("priority", True), ("duration", False))
    if modo == "prioridade":
        return (("priority", True), ("impact", True), ("duration", False))
    return (
        ("priority", True),
        ("impact", True),
        ("duration", False),
    )


def montar_painel_resumo(
    limite: int,
    tempo_utilizado: int,
    impacto_total: int,
    pessoal_pct: float,
    profissional_pct: float,
) -> Panel:
    """Cria painel Rich com os indicadores principais."""
    grid = Table.grid(expand=True)
    grid.add_column(justify="left", style="bold")
    grid.add_column(justify="right", style="bold cyan")
    grid.add_row("Limite diário", f"{limite} min")
    grid.add_row("Tempo utilizado", f"{tempo_utilizado} min")
    grid.add_row("Impacto total", str(impacto_total))
    grid.add_row("Distribuição pessoal", f"{pessoal_pct:.1f}%")
    grid.add_row("Distribuição profissional", f"{profissional_pct:.1f}%")
    return Panel(grid, title="Resumo Diário", border_style="cyan", expand=False)


def imprimir_tabela(
    df: pd.DataFrame,
    titulo: str,
    colunas: Sequence[Tuple[str, str, str]],
) -> None:
    """Imprime um DataFrame como tabela estilizada Rich."""
    if df.empty:
        console.print(f"[yellow]{titulo}: nenhum dado para exibir.[/yellow]")
        return

    table = Table(title=titulo, box=box.MINIMAL_DOUBLE_HEAD, expand=True, show_lines=False)
    for _, header, align in colunas:
        table.add_column(header, justify=align, style="white")

    for _, row in df.iterrows():
        values = []
        for col_key, _, align in colunas:
            value = row[col_key]
            if isinstance(value, float):
                value = f"{value:.1f}" if align != "left" else f"{value:.2f}"
            values.append(str(value))
        table.add_row(*values)

    console.print(table)


def exportar_html(
    df_ordenado: pd.DataFrame,
    tarefas_df: pd.DataFrame,
    cronograma_df: pd.DataFrame,
    caminho: str,
    resumo: Dict[str, float],
) -> None:
    """Gera um relatório HTML simples para compartilhamento."""
    estilo = """
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f6f8fb; color: #0f172a; }
        h1 { color: #2563eb; }
        .card { background: white; padding: 16px 24px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 10px 20px rgba(15,23,42,0.08); }
        table { border-collapse: collapse; width: 100%; margin-top: 12px; }
        th, td { border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }
        th { background: #eff6ff; color: #1d4ed8; }
    </style>
    """
    cards = "".join(
        f"<div><strong>{titulo}</strong><br><span>{valor}</span></div>"
        for titulo, valor in resumo.items()
    )
    html = f"""
    <html>
        <head>
            <meta charset="utf-8" />
            <title>Relatório de Equilíbrio Diário</title>
            {estilo}
        </head>
        <body>
            <h1>Relatório de Equilíbrio Diário</h1>
            <div class="card" style="display:flex; gap:16px; flex-wrap:wrap;">
                {cards}
            </div>
            <div class="card">
                <h2>Tarefas escolhidas</h2>
                {tarefas_df.to_html(index=False)}
            </div>
            <div class="card">
                <h2>Cronograma sugerido</h2>
                {cronograma_df.to_html(index=False)}
            </div>
            <div class="card">
                <h2>DataFrame completo (ordenado)</h2>
                {df_ordenado.to_html(index=False)}
            </div>
        </body>
    </html>
    """
    with open(caminho, "w", encoding="utf-8") as fp:
        fp.write(html)


def merge_sort_dataframe(df: pd.DataFrame, sort_plan: SortPlan, memo=None) -> pd.DataFrame:
    """Ordena um DataFrame usando Merge Sort recursivo com memoização."""
    if memo is None:
        memo = {}

    key = tuple((row.task_id, tuple(getattr(row, col) for col, _ in sort_plan)) for row in df.itertuples())
    if key in memo:
        return memo[key]

    if len(df) <= 1:
        memo[key] = df.copy()
        return memo[key]

    mid = len(df) // 2
    left_sorted = merge_sort_dataframe(df.iloc[:mid].reset_index(drop=True), sort_plan, memo)
    right_sorted = merge_sort_dataframe(df.iloc[mid:].reset_index(drop=True), sort_plan, memo)
    merged = merge_dataframes(left_sorted, right_sorted, sort_plan)
    memo[key] = merged
    return merged


def merge_dataframes(left: pd.DataFrame, right: pd.DataFrame, sort_plan: SortPlan) -> pd.DataFrame:
    """Mescla dois DataFrames já ordenados, respeitando o plano de ordenação."""
    left_records = left.to_dict("records")
    right_records = right.to_dict("records")
    merged_records: List[TaskRecord] = []
    l_idx = r_idx = 0

    while l_idx < len(left_records) and r_idx < len(right_records):
        if precedes(left_records[l_idx], right_records[r_idx], sort_plan):
            merged_records.append(left_records[l_idx])
            l_idx += 1
        else:
            merged_records.append(right_records[r_idx])
            r_idx += 1

    merged_records.extend(left_records[l_idx:])
    merged_records.extend(right_records[r_idx:])
    columns = left.columns if not left.empty else right.columns
    return pd.DataFrame(merged_records, columns=columns)


def precedes(task_a: TaskRecord, task_b: TaskRecord, sort_plan: SortPlan) -> bool:
    """Retorna True se task_a deve vir antes de task_b segundo o plano."""
    for column, desc in sort_plan:
        if task_a[column] == task_b[column]:
            continue

        if desc:
            return task_a[column] > task_b[column]
        return task_a[column] < task_b[column]

    return task_a["task_id"] < task_b["task_id"]


def knapsack_recursive(
    tasks: Sequence[TaskRecord],
    capacity: int,
    index: int = 0,
    memo=None,
    lookup=None,
) -> Tuple[int, Tuple[int, ...]]:
    """Resolve a mochila 0/1 de forma recursiva e com memoização."""
    if memo is None:
        memo = {}
    if lookup is None:
        lookup = {task["task_id"]: task for task in tasks}

    key = (index, capacity)
    if key in memo:
        return memo[key]

    if index >= len(tasks) or capacity <= 0:
        memo[key] = (0, tuple())
        return memo[key]

    current = tasks[index]
    if current["duration"] > capacity:
        result = knapsack_recursive(tasks, capacity, index + 1, memo, lookup)
        memo[key] = result
        return result

    include_value, include_ids = knapsack_recursive(
        tasks, capacity - current["duration"], index + 1, memo, lookup
    )
    include_value += current["impact"]
    include_ids = include_ids + (current["task_id"],)

    exclude_value, exclude_ids = knapsack_recursive(tasks, capacity, index + 1, memo, lookup)

    best = escolher_melhor_solucao(
        (include_value, include_ids),
        (exclude_value, exclude_ids),
        lookup,
    )
    memo[key] = best
    return best


def escolher_melhor_solucao(
    option_a: Tuple[int, Tuple[int, ...]],
    option_b: Tuple[int, Tuple[int, ...]],
    lookup: Dict[int, TaskRecord],
) -> Tuple[int, Tuple[int, ...]]:
    """Desempatador recursivo baseado em impacto, prioridade total e duração."""
    if option_a[0] > option_b[0]:
        return option_a
    if option_b[0] > option_a[0]:
        return option_b

    stats_a = recursao_estatisticas(option_a[1], lookup)
    stats_b = recursao_estatisticas(option_b[1], lookup)

    if stats_a["priority"] > stats_b["priority"]:
        return option_a
    if stats_b["priority"] > stats_a["priority"]:
        return option_b

    if stats_a["duration"] < stats_b["duration"]:
        return option_a
    if stats_b["duration"] < stats_a["duration"]:
        return option_b

    return option_a if option_a[1] <= option_b[1] else option_b


def recursao_estatisticas(
    selection: Tuple[int, ...],
    lookup: Dict[int, TaskRecord],
    idx: int = 0,
    memo=None,
) -> Dict[str, int]:
    """Calcula somas de prioridade e duração de forma recursiva usando memoização."""
    if memo is None:
        memo = {}
    key = (idx, selection[idx:])
    if key in memo:
        return memo[key]

    if idx >= len(selection):
        memo[key] = {"priority": 0, "duration": 0, "impact": 0}
        return memo[key]

    current_task = lookup[selection[idx]]
    tail_stats = recursao_estatisticas(selection, lookup, idx + 1, memo)
    result = {
        "priority": tail_stats["priority"] + current_task["priority"],
        "duration": tail_stats["duration"] + current_task["duration"],
        "impact": tail_stats["impact"] + current_task["impact"],
    }
    memo[key] = result
    return result


def calcular_balanco(
    selection: Tuple[int, ...],
    lookup: Dict[int, TaskRecord],
    idx: int = 0,
    memo=None,
) -> Dict[str, int]:
    """Retorna tempos total/pessoal/profissional de forma recursiva."""
    if memo is None:
        memo = {}
    key = (idx, selection[idx:])
    if key in memo:
        return memo[key]

    if idx >= len(selection):
        memo[key] = {"total": 0, "pessoal": 0, "profissional": 0}
        return memo[key]

    current_task = lookup[selection[idx]]
    partial = calcular_balanco(selection, lookup, idx + 1, memo)
    updated = {
        "total": partial["total"] + current_task["duration"],
        "pessoal": partial["pessoal"] + (current_task["duration"] if current_task["type"] == "pessoal" else 0),
        "profissional": partial["profissional"] + (current_task["duration"] if current_task["type"] == "profissional" else 0),
    }
    memo[key] = updated
    return updated


def formatar_horario(minutos: int) -> str:
    """Converte minutos em HH:MM."""
    horas = minutos // 60
    mins = minutos % 60
    return f"{horas:02d}:{mins:02d}"


def montar_cronograma_recursivo(
    tarefas: Sequence[TaskRecord],
    idx: int = 0,
    inicio: int = 0,
    memo=None,
) -> List[Dict[str, object]]:
    """Gera um cronograma sequencial, respeitando a ordem fornecida."""
    if memo is None:
        memo = {}
    key = (idx, inicio)
    if key in memo:
        return memo[key]

    if idx >= len(tarefas):
        memo[key] = []
        return memo[key]

    atual = tarefas[idx]
    inicio_atual = inicio
    fim_atual = inicio_atual + atual["duration"]
    bloco = {
        "task_id": atual["task_id"],
        "name": atual["name"],
        "start": formatar_horario(inicio_atual),
        "end": formatar_horario(fim_atual),
        "duration": atual["duration"],
        "type": atual["type"],
    }
    restante = montar_cronograma_recursivo(tarefas, idx + 1, fim_atual, memo)
    memo[key] = [bloco, *restante]
    return memo[key]


def produzir_relatorio(
    df: pd.DataFrame,
    selecionados: Tuple[int, ...],
    capacidade: int,
    html_path: str | None = None,
) -> None:
    """Gera o relatório final no terminal e opcionalmente exporta HTML."""
    lookup = {row.task_id: dict(row._asdict()) for row in df.itertuples(index=False)}
    impacto_info = recursao_estatisticas(selecionados, lookup)
    impacto_total = impacto_info["impact"]
    balanco = calcular_balanco(selecionados, lookup)

    tarefas_escolhidas = [lookup[task_id] for task_id in selecionados]
    base_cols = df.columns.tolist()
    tarefas_df = pd.DataFrame(tarefas_escolhidas, columns=base_cols)

    pessoal_pct = (balanco["pessoal"] / balanco["total"] * 100) if balanco["total"] else 0
    profissional_pct = (balanco["profissional"] / balanco["total"] * 100) if balanco["total"] else 0

    console.rule("[bold blue]Relatório de Equilíbrio Diário[/bold blue]")
    console.print(
        montar_painel_resumo(capacidade, balanco["total"], impacto_total, pessoal_pct, profissional_pct)
    )
    console.print()

    console.print("[bold]Tarefas escolhidas (ordenadas por prioridade e impacto)[/bold]")
    if tarefas_df.empty:
        console.print("[yellow]Nenhuma tarefa alcançou o limite de impacto dentro da restrição de tempo.[/yellow]")
        execucao_ordenada: List[TaskRecord] = []
        tarefas_ordenadas_df = pd.DataFrame(columns=df.columns)
    else:
        tarefas_ordenadas_df = tarefas_df.sort_values(by=["priority", "impact", "duration"], ascending=[False, False, True])
        imprimir_tabela(
            tarefas_ordenadas_df,
            "Seleção ótima",
            [
                ("task_id", "ID", "center"),
                ("name", "Tarefa", "left"),
                ("duration", "Duração (min)", "right"),
                ("priority", "Prioridade", "center"),
                ("type", "Tipo", "center"),
                ("impact", "Impacto", "center"),
            ],
        )
        execucao_ordenada = tarefas_ordenadas_df.to_dict("records")

    cronograma = montar_cronograma_recursivo(execucao_ordenada)
    cronograma_df = pd.DataFrame(cronograma)
    console.print("\n[bold]Cronograma sugerido (início/fim em HH:MM)[/bold]")
    if cronograma_df.empty:
        console.print("[yellow]Nenhum cronograma foi montado.[/yellow]")
    else:
        imprimir_tabela(
            cronograma_df,
            "Execução sequencial",
            [
                ("task_id", "ID", "center"),
                ("name", "Tarefa", "left"),
                ("start", "Início", "center"),
                ("end", "Fim", "center"),
                ("duration", "Duração (min)", "right"),
                ("type", "Tipo", "center"),
            ],
        )

    console.print("\n[bold]DataFrame completo ordenado pelo Merge Sort[/bold]")
    imprimir_tabela(
        df,
        "Tarefas (Merge Sort)",
        [
            ("task_id", "ID", "center"),
            ("name", "Tarefa", "left"),
            ("duration", "Duração (min)", "right"),
            ("priority", "Prioridade", "center"),
            ("type", "Tipo", "center"),
            ("impact", "Impacto", "center"),
        ],
    )

    if html_path:
        resumo = {
            "Limite diário": f"{capacidade} min",
            "Tempo utilizado": f"{balanco['total']} min",
            "Impacto total": f"{impacto_total}",
            "Pessoal": f"{pessoal_pct:.1f}%",
            "Profissional": f"{profissional_pct:.1f}%",
        }
        exportar_html(df, tarefas_ordenadas_df, cronograma_df, html_path, resumo)
        console.print(f"\n[green]Relatório HTML salvo em:[/green] {html_path}")


def executar_planejamento(limite_diario: int, modo_ordenacao: str, html_path: str | None = None):
    """Função principal de orquestração."""
    tarefas = gerar_tarefas_base()
    df = pd.DataFrame(tarefas)
    plano_ordenacao = obter_plano_ordenacao(modo_ordenacao)
    df_ordenado = merge_sort_dataframe(df, plano_ordenacao)

    tarefas_ordenadas = df_ordenado.to_dict("records")
    impacto_max, selecionados = knapsack_recursive(tarefas_ordenadas, limite_diario)

    produzir_relatorio(df_ordenado, selecionados, limite_diario, html_path)
    return impacto_max, selecionados


if __name__ == "__main__":
    args = parse_cli_args()
    executar_planejamento(args.limit, args.sort, args.html_report)
