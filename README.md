# Planejador de Equilíbrio Vida-Profissional

Sistema em Python que organiza um dia de trabalho/vida pessoal usando Programação Dinâmica. O script cria um conjunto de tarefas (20+) com duração, prioridade, tipo e impacto no bem-estar. Em seguida, ordena os dados com Merge Sort recursivo (com memoização) e aplica Mochila 0/1 também recursiva para escolher automaticamente a combinação de tarefas que maximiza o impacto sem ultrapassar o limite diário.

## Requisitos

- Python 3.12+
- Pandas
- Rich

No Ubuntu/WSL:

```bash
sudo apt-get install -y python3-pandas python3-rich
```

No Windows:

```powershell
py -m pip install pandas rich
```

## Como rodar

```bash
python3 planner.py \
  [--limit MINUTOS] \
  [--sort hibrido|impacto|prioridade] \
  [--html-report caminho.html] \
  [--stress-log stress_history.json]
```

Exemplos:

- `python3 planner.py` → usa limite padrão de 480 min e ordenação híbrida (prioridade > impacto > duração).
- `python3 planner.py --limit 420 --sort impacto` → reorganiza os dados priorizando impacto e aplica o novo limite antes da mochila.
- `python3 planner.py --html-report relatorio.html` → exporta um arquivo HTML estilizado além do relatório no terminal.
- `python3 planner.py --stress-log ""` → desativa o registro do histórico semanal (padrão grava em `stress_history.json` para acompanhar a evolução).

O script imprime um painel formatado (via Rich) com as tarefas escolhidas, tempo total utilizado, impacto alcançado, equilíbrio pessoal/profissional, cronograma sequencial, monitor de estresse diário/semanal e o DataFrame ordenado (incluindo a nova coluna “Perfil stress” ao lado do impacto). Opcionalmente, tudo é espelhado em HTML.

## Como a solução funciona

1. **DataFrame base** – As tarefas são definidas em `gerar_tarefas_base()` e convertidas para um `DataFrame` do pandas.
2. **Ordenação com Merge Sort** – A função `merge_sort_dataframe` divide recursivamente o DataFrame em metades, ordena cada metade (memoizando subproblemas iguais) e mescla seguindo o plano (prioridade e impacto decrescente, duração crescente).
3. **Mochila 0/1 recursiva** – `knapsack_recursive` utiliza programação dinâmica top-down (memoização). O “peso” é a duração, o “valor” é o impacto. Cada decisão considera incluir ou excluir a tarefa atual.
4. **Processamento adicional recursivo** – Funções `recursao_estatisticas`, `calcular_balanco` e `montar_cronograma_recursivo` percorrem recursivamente a lista de tarefas escolhidas para calcular somatórios, equilíbrio e montar um cronograma HH:MM.
5. **Monitor de estresse** – `calcular_estresse_diario` (recursivo) avalia cada impacto/tipo de tarefa, transforma o somatório para a escala 0–15 (quanto maior, pior) e gera o status diário; `avaliar_estresse_semanal` usa memoização para produzir médias históricas a partir do log JSON e aplica um baseline para diferenciar o status semanal. Cada tarefa recebe o rótulo “Perfil stress” segundo seu impacto: impacto 3 pessoal = Recuperador, impacto 3 profissional = Produtivo, impacto 2 pessoal = Equilibrado, demais = Estressante.
6. **Relatório** – `produzir_relatorio` usa Rich para montar um painel com indicadores, tabelas legíveis, cronograma, perfis de estresse por tarefa e monitor diário/semanal; também exporta HTML quando `--html-report` é informado.

### Conceitos de Programação Dinâmica usados

- **Subestruturas ótimas**: tanto na ordenação quanto na mochila, o resultado global depende da combinação ótima de subproblemas.
- **Sobreposição de subproblemas**: chamadas com os mesmos parâmetros são armazenadas em dicionários de memoização para evitar recomputações.
- **Recursão + memoização (top-down)**: todas as funções principais trabalham dessa forma, atendendo ao requisito da disciplina.

## Exemplo de saída

```text
──────────────────────── Relatório de Equilíbrio Diário ────────────────────────
╭───────── Resumo Diário ──────────╮      ╭──────────────────────────── Monitor de Estresse ─────────────────────────────╮
│ Limite diário            480 min │      │ Status diário   Alerta moderado · Pontuação 7.7                              │
│ Tempo utilizado          465 min │      │                 Inclua pausas ou atividades pessoais de impacto 3.           │
│ Impacto total                 31 │      │ Status semanal  Alerta moderado · Pontuação 6.4                              │
│ Distribuição pessoal       59.1% │      │                 Escala 0-15 · Quanto maior, maior o estresse acumulado       │
│ Distribuição profissional  40.9% │      ╰──────────────────────────────────────────────────────────────────────────────╯
╰──────────────────────────────────╯

Tarefas escolhidas (ordenadas por prioridade e impacto)
                                 Seleção ótima                                  
     ╷                   ╷               ╷            ╷              ╷         ╷
  ID │ Tarefa            │ Duração (min) │ Prioridade │     Tipo     │ Impacto │ Perfil stress
 ════╪═══════════════════╪═══════════════╪════════════╪══════════════╪═════════╪═══════════════
  1  │ Meditação guiada  │            20 │     5      │   pessoal    │    3    │ Recuperador
  3  │ Revisão de metas  │            30 │     5      │ profissional │    3    │ Produtivo
 16  │ Planejamento ...  │            35 │     4      │   pessoal    │    3    │ Recuperador
    ...

Cronograma sugerido (início/fim em HH:MM)
                              Execução sequencial                               
     ╷                         ╷        ╷       ╷               ╷               
  ID │ Tarefa                  │ Início │  Fim  │ Duração (min) │     Tipo      
 ════╪═════════════════════════╪════════╪═══════╪═══════════════╪══════════════ 
  1  │ Meditação guiada        │ 00:00  │ 00:20 │            20 │   pessoal     
  3  │ Revisão de metas        │ 00:20  │ 00:50 │            30 │ profissional  
    ...

DataFrame completo ordenado pelo Merge Sort
                              Tarefas (Merge Sort)                              
     ╷                   ╷               ╷            ╷              ╷          
  ID │ Tarefa            │ Duração (min) │ Prioridade │     Tipo     │ Impacto  
 ════╪═══════════════════╪═══════════════╪════════════╪══════════════╪═════════ 
  1  │ Meditação guiada  │            20 │     5      │   pessoal    │    3     
  3  │ Revisão de metas  │            30 │     5      │ profissional │    3     
    ...
```

> **Escala de impacto e estresse**
>
> - Impacto 3 pessoal → Recuperador; impacto 3 profissional → Produtivo  
> - Impacto 2 pessoal → Equilibrado; demais combinações → Estressante  
> - O monitor converte o dia para a escala 0–15. Pontuações até 6 indicam equilíbrio, de 7–10 exigem atenção moderada e acima de 10 representam risco alto.

## Estrutura do repositório

- `planner.py` – Implementação completa com tarefas, ordenação recursiva, mochila DP e relatório.
- `README.md` – Este arquivo com documentação acadêmica/pedagógica.
- `stress_history.json` – Arquivo criado automaticamente (opcional) para armazenar o histórico das pontuações diárias e permitir o status semanal.
