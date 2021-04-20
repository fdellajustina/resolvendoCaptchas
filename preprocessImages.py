import os
import copy
from cv2 import imread, imshow, waitKey, destroyAllWindows, cvtColor, COLOR_BGR2GRAY
import numpy as np
from string import ascii_lowercase
from sklearn.model_selection import train_test_split

class PreprocessImages:

    def __init__(self):

        self.classes_captchSimbols = ascii_lowercase + '0123456789'
        self.set_split_letters_on_captcha_images()  # x_dataset
        self.set_targetVariables_oneHotEncoder()    # y_dataset

    def set_images_names_from_filesName(self, verbose=False):

        self.images_names_list = os.listdir(
            os.path.join(os.getcwd(), "samples")
        )

        if verbose:
            print('\n Some images names: \n', self.images_names_list[10:40])

        self.original_dataset_size = len(self.images_names_list)

    def get_all_images_on_folder_list(self, verbose=False, showImage=False):

        self.set_images_names_from_filesName()

        images = [
            imread(os.path.join(os.getcwd(), f"samples/{imgName_string}"))
            for imgName_string in self.images_names_list
        ]

        if verbose:
            print(f"\nTotal de figuras no dataset: {len(images)}")
            print(f"Shape das imagens do dataset: {images[0].shape}")

        if showImage:
            imshow("Image 0 on folder", images[0])
            waitKey(0)
            destroyAllWindows()

        return images

    def get_imageNormalized_grayScale(self, verbose=False):

        images = self.get_all_images_on_folder_list()

        gray_images = [cvtColor(img, COLOR_BGR2GRAY) for img in images]

        normalizedImg = [img/255 for img in gray_images]

        if verbose:
            print(f"\nShape das imagens em escala de cinza do dataset: {gray_images[0].shape}")
            print(f'Images scale: {min(normalizedImg), max(normalizedImg)}')

        return normalizedImg

    def set_split_letters_on_captcha_images(self, verbose=False):
        '''
        Para treinar a rede, ao inves de entregarmos a imagem completa, faremos recortes para fazer com que ela reconhça caractere por caractere da imagem comleta.
        O motivo por fazermos assim é porque por exemplo para a rede reconhecer "2b827" temos apenas uma imagem no dataset, portanto a rede não conseguiria aprender essa combinação de maneira eficiente bem como não conseguiriamos obter todas as combinações possíveis de caracteres e número num dataset de 1070 imagens.
        Cada imagem contem 5 caractere, portanto nosso novo dataset terá 5*len(datasetOld).
        Verificou-se que cada caractere fica espaçado 20 pixels, portanto faremos recortes na imagem original iniciando em 30 e dando steps de 20
        Para as alturas pegaremos sempre no intervalo de [12:50]. Imagem original vai de [0:50] em y, só iremos recortar uma parte superior ao iniciar em 12.
        Principal problema desse implementação é com a letra m que é um pouco maior e ao usarmos janelas de 20 em 20 em x acabamos recortando o m ou deixando de pegar a última letra.
        :return:
        '''

        gray_images = self.get_imageNormalized_grayScale()

        x_new = np.zeros((self.original_dataset_size * 5, 38, 20))
        start_x, start_y, width, end_y = 30, 12, 20, 50
        for img_idx, grayImg in enumerate(gray_images):
            px = start_x
            for character_idx in range(5):
                x_new[img_idx * 5 + character_idx] = grayImg[start_y:end_y, px:px + width]
                px += width

        if verbose:
            print(f"\nTotal de figuras no dataset: {len(x_new)}")
            print(f"Shape das imagens do dataset: {x_new[0].shape}")

        ######################################
        # Preparando o dataset para a rede neural. Iremos normalizar a escala da imagem para ir de 0 a 1
        # precisamos também passar para a rede neural um dataset com a dimensão (n_amostras, (dimensão imagem), gray_scale)
        self.x_dataset = np.zeros((x_new.shape[0], x_new.shape[1], x_new.shape[2], 1))
        for idx, img in enumerate(x_new):
            self.x_dataset[idx] = np.reshape(img, (x_new.shape[1], x_new.shape[2], 1))

        if verbose:
            print(f"\nTotal de figuras no dataset: {len(self.x_dataset)}")
            print(f"Shape das imagens do dataset: {self.x_dataset[0].shape}")

    def get_images_name_without_extension(self):
        '''
        Obtem o nome da imagem *.png, retirando a extensão "png".
        :return:
        '''

        # primeiramente vamos pegar só o nome da imagem excluindo o ".png"
        y_tmp = copy.deepcopy(self.images_names_list)
        y_tmp = [y[0:5] for y in y_tmp]
        captchas = [letter[j] for _, letter in enumerate(y_tmp) for j in range(5)]

        return captchas

    def set_targetVariables_oneHotEncoder(self, verbose=True):
        '''
        O nome da imagem coincidentemente está organizada de tal forma que a seguencia representa exatamente a seguencia dos caracteres contidos nas imagens.
        Portanto as variaveis target serão definidas com base no nome dos arquivos
        As variaveis target, isto é, as classes são 36 no total (26 letras do alfabeto e mais 10 representando os números de 0-9.
        :return:
        '''

        captchas = self.get_images_name_without_extension()

        self.y_dataset = np.zeros((len(captchas), 36), dtype='uint8')
        for y_idx, letter in enumerate(captchas):
            # retorna o indice onde está a "letter" (mostra sempre o primeiro indice, se houver outros f aparecerá o indice só do primeiro)
            loc_character = self.classes_captchSimbols.find(letter)
            self.y_dataset[y_idx, loc_character] = 1

        if verbose:
            print(f'\nTarget variables shape: {self.y_dataset.shape}')

    def get_train_and_test_subset(self, verbose=True):

        x_train, x_test, y_train, y_test = train_test_split(
            self.x_dataset,
            self.y_dataset,
            test_size=0.2,
            random_state=8
        )

        if verbose:
            print(f'\nShape dos dados de treinameno: {x_train.shape}')
            print(f'Shape dos dados de teste: {x_test.shape}')

        return x_train, x_test, y_train, y_test