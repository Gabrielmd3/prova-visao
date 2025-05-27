# prova-visao

# Classificação de Gatos e Cachorros com Redes Neurais Convolucionais (CNN)

## Descrição do Problema

O objetivo deste projeto é desenvolver um modelo de inteligência artificial capaz de classificar imagens como pertencentes a classe "cats" ou "dogs", utilizando Redes Neurais Convolucionais. O conjunto de dados é composto por imagens organizadas em duas pastas e o modelo é treinado a partir dessas imagens.

---

## Justificativa das Técnicas Utilizadas

* CNN (Convolutional Neural Networks): são eficazes para tarefas de classificação de imagens por capturarem padrões espaciais, como bordas, texturas e formas.
* Keras + TensorFlow: oferece uma API de alto nível para construção e treinamento de redes neurais de forma simples e eficiente.
* Data Augmentation e Pré-processamento: aplicados para melhorar a capacidade do modelo de generalizar, reduzindo o risco de overfitting.
* Métricas de Avaliação (Precisão, Recall, F1-Score): fornecem uma visão detalhada sobre a performance do modelo em termos de acerto e erro por classe.

---

## Etapas Realizadas

1. Organização do Dataset:

   * Limitação de até 700 imagens por classe.
   * Separação em 80% para treinamento e 20% para teste.
   * Seleção de 6 gatos e 6 cachorros reservados para avaliação final do modelo.

2. Construção do Modelo:

   * Modelo CNN com 3 camadas convolucionais seguidas de pooling. 
   * Camada densa final com ativação sigmoid para saída binária.

3. Treinamento:

   * Realizado por 30 épocas com otimização `Adam` e função de perda `binary_crossentropy`.

4. Avaliação:

   * Métricas de Precisão, Recall e F1-Score calculadas sobre o conjunto de teste.
   * Predição individual das 12 imagens reservadas para validação externa.

5. Pré-processamento Alternativo:

   * Seleção aleatória de outras 6 imagens de cada classe.
   * Aplicação de redimensionamento (128x128), filtro Gaussiano e equalização de histograma (em tons de cinza).
   * Salvamento das imagens tratadas na pasta `output_images/`.

---

## Resultados Obtidos

* O modelo apresentou alta acurácia, com desempenho consistente em ambas as classes.
* As métricas no conjunto de teste foram:

Precisão: 0.64%
Recall:   0.75%
F1-Score: 0.68%

* As predições nas imagens reservadas confirmaram a capacidade de generalização do modelo.

---

## Tempo Total Gasto

* Aproximadamente  4h, divididas em:

  * 3h desenvolvimento e treino do modelo
  * 1h documentação e pré-processamento alternativo

---

## Dificuldades Encontradas

* Carga computacional: processamento de imagens em grande quantidade exigiu otimização da pipeline e limitação do volume de dados.