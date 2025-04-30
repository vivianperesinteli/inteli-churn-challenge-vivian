# inteli-churn-challenge-vivian

# Desafio de Previsão de Churn da Inteli Academy da Vivs Peres

## Visão Geral
Este projeto prevê o churn de clientes para a TelecomPlus usando Regressão Logística e XGBoost, alcançando 70% de acurácia via validação cruzada.

## Conjunto de Dados
- **Treino**: `dados_clientes.csv` (features e rótulos de churn)
- **Teste**: `desafio.csv` (apenas features)
- **Saída**: `resultado_yourname_yoursurname.csv` (previsões)

## Abordagem
1. **Exploração**: Analisou distribuição de dados e correlações.
2. **Pré-processamento**: Tratou valores ausentes, codificou `produtos_assinados` com `MultiLabelBinarizer` e normalizou features numéricas.
3. **Modelagem**: Treinou Regressão Logística e XGBoost, selecionou o melhor via validação cruzada.
4. **Importância de Features**: Identificou drivers principais (por exemplo, tenure).
5. **Submissão**: Previu churn em `desafio.csv`.

## Insights
- Baixo tempo de contrato e assinaturas específicas impulsionam o churn.
- XGBoost superou a Regressão Logística devido a padrões não lineares.

## Requisitos
- Python 3.9+
- Instalar: `pip install -r requirements.txt`

## Como Executar
1. Clonar: `git clone https://github.com/yourusername/inteli-churn-challenge.git`
2. Colocar `dados_clientes.csv` e `desafio.csv` em `data/`.
3. Executar `churn_prediction.ipynb` no Jupyter Notebook.
4. Saída: `resultado_yourname_yoursurname.csv`

## Arquivos
- `churn_prediction.ipynb`: Notebook principal
- `data/resultado_yourname_yoursurname.csv`: Previsões
- `data/*.joblib`: Modelos/preprocessadores salvos
- `requirements.txt`: Dependências
- `*.png`: Visualizações
