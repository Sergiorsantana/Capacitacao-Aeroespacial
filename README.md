# Capacitacao-Aeroespacial
Repositório referente a Capacitação e Residência Tecnológica em Tecnologias Aeroespaciais Avançadas com o Instituto de Hardware HBR 

### Trabalho Final Capacitação e Residência Tecnológica em Tecnologias Aeroespaciais Avançadas

<img src="https://hardware.org.br/capacitacao/aero/images/item/item9.webp" alt="Imagem de Exemplo da Fuselagem" width="300"/>

A Capacitação e Residência Tecnológica em Tecnologias Aeroespaciais Avançadas destaca os temas de inteligência artificial, segurança cibernética para aviação e biotecnologia para a área aeroespacial e tem duração de 10 meses. Participarão da Capacitação Básica, durante os três primeiros meses, 120 ingressantes, 40 para cada disciplina. Desses, 30 alunos participarão da fase de Capacitação Avançada de quatro meses, seguida da Residência Tecnológica de três meses em instituições parceiras.

O principal objetivo da Residência Tecnológica em Tecnologias Aeroespaciais Avançadas é a capacitação de profissionais para a área aeroespacial brasileira, principalmente a do Estado de São Paulo, tendo como principal diferencial a relevância dos temas propostos para o setor e a qualidade da capacitação oferecida.




**Aluno: Isaias Abner Lima Saraiva**

## Fonte dos Dados e Estrutura dos Dados

Para o treinamento dos modelos, será utilizado o *dataset* **"Aircraft Damage Detection"** (obtido via [Roboflow Universe](https://universe.roboflow.com/college-jcb9y/aircraft-damage-detection-a8z4k)).

### Detecção de Defeitos

O *dataset* original é rotulado para **Detecção de Objetos** (localização da falha via *bounding boxes*). O foco é treinar modelos para **detecção de defeitos em aeronaves**, ou seja, identificar a presença e a localização das anomalias.

| Categoria de Classe | Condição Original do Dataset | Rótulo de Detecção |
| :--- | :--- | :--- |
| **DEFECTO** (Classe 1) | Imagens que possuem **anotações de defeitos** (presença de *bounding boxes*). | A imagem **contém** um defeito. |
| **NORMAL** (Classe 0) | Imagens que **não possuem anotações** (imagens de fundo limpo). | A imagem está **sem falhas**. |

