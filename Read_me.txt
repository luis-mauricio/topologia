Segue em anexo o programa para análise topológica de lineamentos expedita com uma interface meia-bomba. É em Matlab, precisa do sistema, claro.

O arquivo de entrada (exemplo Frattopo.xyz) é o tradicional xi yi xf yf ASCII. Nos diagramas triangulares, a região em magenta delimita o campo "geológico" (pelo menos o que consegui com umas 60 análises em áreas diversas). Os valores para complexidade geológica são divididos em CB 1 e 1,3, como faço sempre.

Precisa, além do arquivo, dos parâmetros: data.vizinhanca, o raio para buscar as conexões (valor depende das coordenadas usadas) e data.discretizacao, número de polígonos para cálculo do CB e geração do mapa de calor.
