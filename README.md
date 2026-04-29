# Pesquisa em Visao Computacional para Tecnologia Assistiva

Este projeto tem como objetivo estudar conceitos de visao computacional, aprendizado de maquina e deteccao de objetos, aplicando esses conhecimentos em um contexto de tecnologia assistiva para pessoas com deficiencia visual.

A proposta da pesquisa e investigar como modelos de deteccao podem reconhecer cedulas de dinheiro em imagens, permitindo que esse tipo de solucao possa evoluir futuramente para ferramentas mais acessiveis, como aplicativos com retorno por voz, interfaces simplificadas ou sistemas embarcados de apoio a autonomia.

## Objetivo da Pesquisa

- Entender o funcionamento de datasets anotados para deteccao de objetos.
- Preparar imagens e labels no formato esperado pelo YOLO.
- Treinar um modelo para reconhecer diferentes cedulas.
- Avaliar resultados de deteccao em imagens reais.
- Explorar possibilidades de aplicacao em tecnologias assistivas para pessoas com deficiencia visual.

## Contexto Assistivo

Pessoas com deficiencia visual podem enfrentar dificuldades para identificar cedulas, objetos, textos e informacoes visuais no dia a dia. A deteccao automatizada por imagem pode apoiar a criacao de recursos que aumentem a independencia, desde que sejam pensados com acessibilidade, confiabilidade e facilidade de uso.

Neste projeto, o reconhecimento de cedulas funciona como um estudo inicial para compreender os desafios tecnicos envolvidos, como qualidade das imagens, variacao de iluminacao, posicionamento dos objetos, confianca das predicoes e apresentacao do resultado ao usuario.

## Tecnologias Utilizadas

- Python
- Ultralytics YOLO
- OpenCV
- PySide6
- Dataset anotado no formato YOLO

## Estrutura do Projeto

```text
.
├── dataset/cedulas/        # Dataset preparado para treino e validacao
├── runs/cedulas_yolo11n/   # Resultados do treinamento e pesos gerados
├── script.py               # Prepara o dataset no formato YOLO
├── train_yolo.py           # Treina o modelo YOLO para deteccao de cedulas
├── detect_ui.py            # Interface grafica para testar deteccoes em imagens
├── yolo11n.pt              # Modelo base usado no treinamento
└── README.md
```

## Classes do Dataset

O dataset configurado em `dataset/cedulas/data.yaml` possui as seguintes classes:

- Cedula 5
- Cedula 10
- Cedula 20
- Cedula 50
- Cedula 100
- Cedula 200

## Como Executar

### 1. Instalar dependencias

Com o ambiente virtual ativado, instale as bibliotecas principais:

```bash
pip install ultralytics opencv-python PySide6
```

### 2. Preparar o dataset

O arquivo `script.py` organiza imagens e labels em pastas de treino e validacao:

```bash
python script.py
```

Observacao: o script usa caminhos locais definidos nas constantes `DATASET_ROOT` e `EXPORT_ROOT`. Caso o dataset esteja em outro local, esses caminhos devem ser ajustados antes da execucao.

### 3. Treinar o modelo

```bash
python train_yolo.py
```

O treinamento utiliza o modelo base `yolo11n.pt` e salva os resultados em `runs/cedulas_yolo11n/`.

### 4. Testar deteccoes pela interface

```bash
python detect_ui.py
```

A interface permite selecionar uma imagem, executar a deteccao e visualizar as cedulas identificadas com suas respectivas confiancas.

## Resultados

Os resultados do treinamento ficam em `runs/cedulas_yolo11n/`, incluindo graficos de desempenho, matriz de confusao, exemplos de predicao e os pesos do modelo:

- `weights/best.pt`: melhor modelo gerado durante o treinamento.
- `weights/last.pt`: ultimo modelo salvo.
- `results.png`: resumo das metricas de treinamento.
- `confusion_matrix.png`: matriz de confusao.

## Possiveis Evolucoes

- Adicionar retorno por voz para informar a cedula detectada.
- Melhorar a interface para uso com leitores de tela.
- Testar o modelo com imagens capturadas por camera em tempo real.
- Expandir o dataset com mais exemplos, angulos, iluminacoes e fundos.
- Avaliar o uso em dispositivos moveis ou sistemas embarcados.
- Medir a confiabilidade do modelo em cenarios reais de uso.

## Observacao

Este projeto tem finalidade de estudo e pesquisa. Para uso assistivo real, seria necessario validar a solucao com criterios de seguranca, acessibilidade, privacidade e testes com usuarios, evitando que predicoes incorretas prejudiquem a tomada de decisao.
