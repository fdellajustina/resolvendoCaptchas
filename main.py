import os
import cv2
import numpy as np
import copy

images_names = os.listdir(os.path.join(os.getcwd(), "samples"))
# print(images_names)

###############################################
# carregando todas as imagens da pasta
images = [cv2.imread(os.path.join(os.getcwd(), "samples/{}".format(imgName_string)))
          for imgName_string in images_names]

print(f"\nTotal de figuras no dataset: {len(images)}")
print(f"Shape das imagens do dataset: {images[0].shape}")

# # mostrando uma imagem
# cv2.imshow("Captcha 0", images[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################################
# testando se imagens estão em escal RGB e não cinza
# (b, g, r) = cv2.split(images[0])
# maskGrayTest = np.zeros(images[0].shape[:2], dtype='uint8')
# cv2.imshow("Vermelho", cv2.merge([maskGrayTest, maskGrayTest, r]))
# cv2.imshow("Verde", cv2.merge([maskGrayTest, g, maskGrayTest]))
# cv2.imshow("Azul", cv2.merge([b, maskGrayTest, maskGrayTest]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###############################################
# convertendo imagens para escala de cinza
gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
print(f"\nShape das imagens em escala de cinza do dataset: {gray_images[0].shape}")

###############################################
# Para treinar a rede, ao inves de entregarmos a imagem completa, faremos recortes para fazer com que ela reconhça caractere por caractere da imagem comleta.
# O motivo por fazermos assim é porque por exemplo para a rede reconhecer "2b827" temos apenas uma imagem no dataset, portanto a rede não conseguiria aprender essa combinação de maneira eficiente bem como não conseguiriamos obter todas as combinações possíveis de caracteres e número num dataset de 1070 imagens.
# cada imagem contem 5 caractere, portanto nosso novo dataset terá 5*len(datasetOld).
# verificou-se que cada caractere fica espaçado 20 pixels, portanto faremos recortes na imagem original iniciando em 30 e dando steps de 20
# para as alturas pegaremos sempre no intervalo de [12:50]. Imagem original vai de [0:50] em y, só iremos recortar uma parte superior ao iniciar em 12
# principal problema desse implementação é com a letra m que é um pouco maior e ao usarmos janelas de 20 em 20 em x acabamos recortando o m ou deixando de pegar a última letra
x_new = np.zeros((len(images_names)*5, 38, 20))
start_x, start_y, width, end_y = 30, 12, 20, 50
for img_idx, grayImg in enumerate(gray_images):
    px = start_x
    for character_idx in range(5):
        x_new[img_idx*5+character_idx] = grayImg[start_y:end_y, px:px+width]
        px += width

print(f"\nTotal de figuras no dataset: {len(x_new)}")
print(f"Shape das imagens do dataset: {x_new[0].shape}")

###############################################
# Preparando o dataset para a rede neural. Iremos normalizar a escala da imagem para ir de 0 a 1
# precisamos também passar para a rede neural um dataset com a dimensão (n_amostras, (dimensão imagem), gray_scale)
x_dataset = np.zeros((x_new.shape[0], x_new.shape[1], x_new.shape[2], 1))
for idx, img in enumerate(x_new):
    norm = img/250  # normalizando os dados
    x_dataset[idx] = np.reshape(norm, (x_new.shape[1], x_new.shape[2], 1))

print(f"\nTotal de figuras no dataset: {len(x_dataset)}")
print(f"Shape das imagens do dataset: {x_dataset[0].shape}")

###############################################
# criando as variável target (as labels) de cada imagem. Para isso vamos usar o nome da imagem *.png que coincidentemente está organizada de tal forma que a seguencia representa exatamente a seguencia dos caracteres contidos nas imagens

# primeiramente vamos pegar só o nome da imagem excluindo o ".png"
y_actual = copy.deepcopy(images_names)
y_actual = [y[0:5] for y in y_actual]
y = [letter[j] for _, letter in enumerate(y_actual) for j in range(5)]

## verificando se y corresponde mesmo a image x_dataset
# i = 100
# cv2.imshow("x0", x_dataset[i])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(y[i])

###############################################
# Fazendo OneHotEncode na variavel y. Para tanto vamos definir todas as possíveis classes agrupando todos os simbolos possíveis em um unico string
import string
simbols = string.ascii_lowercase + '0123456789'

y_dataset = np.zeros((len(y), 36), dtype='uint8')
for y_idx, letter in enumerate(y):
    loc_character = simbols.find('f') # retorna o indice onde temos a letra f (mostra sempre o primeiro indice, se houver outros f aparecerá o indice só do primeiro)
    y_dataset[y_idx, loc_character] = 1

print('\n', y_dataset.shape, y_dataset[0])

###############################################
# separando dados de treino e teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=8)
print(x_train.shape)

