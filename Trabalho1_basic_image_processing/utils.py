"""Funções utilizadas no trabalho 1 de MO443
Luís Antônio Almeida Lima Vieira - RA 221045
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Any
import time


def plot(
    image: np.array,
    function: callable,
    config: dict={},
    title: str = "",
    figsize: tuple = (15, 5),
    n_plots: tuple = (1, 3),
    cmap: str = "gray",
    show_axis: bool = False,
    show_original: bool = True,
    save: bool = False,
    save_dir: str = 'outputs/',
    border_zoom: bool = False,
) -> None:
    """
    Aplica transformações a uma imagem recebendo uma função na variável
    function, salva as imagens transformadas e exibe os resultados
    em um gráfico com subplots.
    Parâmetros de Parâmetros de entrada:
        - image: Imagem original.
        - function: Função a ser aplicada na imagem.
        - config: Dicionário para variar de parâmetros da função.
            Formato: {
                "titulo do subplot 1": {"parametro1": "valor", ...},
                "titulo do subplot 2": {"parametro1": "valor", ...},
                ...
            }
        - title: Título geral do gráfico.
        - figsize: Tamanho da figura.
        - n_plots: Número de (linhas, colunas) dos subplots.
        - cmap: Mapa de cores das imagens.
            Ex gray, bgr (blue, green, red) ou rgb (red, green, blue).
        - show_axis: Se True, mostra os eixos dos plots.
        - show_original: Se True mostra a imagem original em um subplot
        - save: Se True salva os plots em save_dir.
        - save_dir: Diretório para salvar as imagens.
        - border_zoom: Se True, aplica um zoom no canto infeiror da imagem
    Saída da função: None.
    """
    fig, axs = plt.subplots(n_plots[0], n_plots[1], figsize=figsize)
    axs = axs.flatten()
    # plot da imagem original
    if show_original:
        # Tratamento para diferentes tipos de canais nas imagens
        if cmap == "gray": # gray
            axs[0].imshow(image, cmap="gray")
        elif cmap == "bgr": # (blue, green, red)
            axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else: # (red, green, blue) e demais casos
            axs[0].imshow(image)
        # Define o título e se os eixos aparecerão
        axs[0].set_title('Imagem original', fontsize=16)
        axs[0].axis(show_axis)

    # plot das imagens transformadas
    for i, (key, value) in enumerate(config.items()):
        # Define o índice do próximo subplot
        index = i if not show_original else i + 1
        # Aplica a função, com a imagem e parâmetros definidos em config
        transformed_image = function(image, **value)
        # Tratamento para diferentes tipos de canais nas imagens
        if cmap == "gray": # gray
            axs[index].imshow(transformed_image, cmap="gray")
        elif cmap == "bgr": # (blue, green, red)
            axs[index].imshow(
                cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            )
        else: #(red, green, blue) e demais casos
            axs[index].imshow(transformed_image)
        # Define o título e se os eixos aparecerão
        axs[index].set_title(key, fontsize=16)
        axs[index].axis(show_axis)
        # Aplica um zoom nas bordas se necessário
        if border_zoom:
            axs[index].set_xlim(0, 10)
            axs[index].set_ylim(0, 10)
        # Salva a imagem transformada
        if save:
            # Nome do arquivo será o título geral do gráfico
            parameter = ""
            if value.values():
                parameter = list(value.values())[0]
                # Se o primeiro parâmetro for um número ou string,
                # entra no nome do arquivo
                # Caso contrário, o nome do arquivo vai com o nome do subplot
                if not isinstance(parameter, (str, int, float)) and key:
                    parameter = key.replace(" ", "_").lower()
            cv2.imwrite(
                f'{save_dir}/{title}_{parameter}.png',
                transformed_image
            )

    # Omite os subplots não utilizados (casos de número ímpar de plots)
    n_used = len(config) + (1 if show_original else 0)
    for i in range(n_used, len(axs)):
        axs[i].axis('off')

    # Salva a figura para completa para o relatório e mostra o plot
    plt.tight_layout()
    if save:
        plt.savefig(f'{save_dir}/{title}.png', bbox_inches='tight', dpi=300)
    plt.show()


def rotate_image(image: np.array, angle: int) -> np.array:
    """
    Rotaciona imagens monocromáticas em ângulos múltiplos de 90 graus.
    Parâmetros de entrada:
        - image: Imagem monocromática.
        - angle: Ângulo de rotação (múltiplo de 90 graus).
    Saída da função: Imagem rotacionada.
    """
    # Tratamento para casos de ângulos de entrada fora do intervalo [0, 360]
    # Transforma para os ânguloes equivalentes 90, 180, 270 e 0
    angle %= 360

    # Para 90 graus, basta transpor a imagem e inverter as colunas
    if angle == 90:
        return image.T[:, ::-1]
    # Para 180 graus, basta inverter as linhas e as colunas
    elif angle == 180:
        return image[::-1, ::-1]
    # Para 270 graus, basta transpor a imagem e inverter as linhas
    elif angle == 270:
        return image.T[::-1, :]
    # Para 0 ou 360 graus, retorna a imagem original
    else:
        return image


def expansion(image: np.array, factor: int) -> np.array:
    """
    Aumenta a escala da imagem monocromática pela replicação de pixels
    Parâmetros de entrada:
        - image: Imagem monocromática.
        - factor: Fator de ampliação (maior que 1 para ampliar).
    Saída da função: Imagem ampliada pelo fator especificado.
    OBS:
        Se o fator for menor ou igual a 1, retorna a imagem original.
    """
    # Se o fator for menor ou igual a 1, retorna a imagem original
    if factor <= 1:
        return image

    # Replicação dos pixels nos eixos linha e coluna pela função np.repeat
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)


def pencil_effect(image: np.array, kernel_size: int = 21) -> np.array:
    """
    Aplica um efeito de desenho a lápis na imagem.
    Parâmetros de entrada:
        - image: Imagem original.
        - kernel_size: Tamanho do filtro gaussiano
    Saída da função:
        Imagem transformada com efeito de desenho a lápis.
    OBS:
        O tamanho do filtro kernel_size deve ser ímpar para funcionar.
    """
    # Tamanho do filtro deve ser ímpar para garantir um centro definido
    # na convolução.
    if kernel_size % 2 == 0:
        raise ValueError("Kernel precisa ser ímpar!")

    # Converte a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplica um filtro gaussiano de desfoque para suavizar a imagem
    # Parâmetros de desvio padrão sigmax e sigmay são colocados em 0
    # Com isso são calculados automaticamente com base no tamanho do kernel
    blurred_image = cv2.GaussianBlur(
        gray_image,
        (kernel_size, kernel_size),
        0
    )

    # Divide a imagem em escala de cinza pela imagem desfocada.
    # Reescala, multiplicando por 256, e limitar os valores em [0, 255]
    # Multiplica por 256 para mapear valores próximas de 1 para serem 255.
    # Divisão por (blurred_image + 1e-5) para evitar divisão por zero
    # Resultado final é convertido para uint8, que é padrão na imagem original
    pencil_image = np.clip(
        (gray_image / (blurred_image + 1e-5)*256), 0, 255
    ).astype(np.uint8)

    return pencil_image


def pencil_effect_cv(image: np.array, kernel_size: int = 21) -> np.array:
    """
    Efeito de desenho a lápis com funcões prontas do Opencv para divisão.
    Parâmetros de entrada:
        - image: Imagem original.
        - kernel_size: Tamanho do filtro gaussiano
    Saída da função:
        Imagem transformada com efeito de desenho a lápis.
    """
    # Tamanho do filtro deve ser ímpar para garantir um centro definido
    # na convolução.
    if kernel_size % 2 == 0:
        raise ValueError("Kernel precisa ser ímpar!")

    # Converte a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplica um filtro gaussiano de desfoque para suavizar a imagem
    # Parâmetros de desvio padrão sigmax e sigmay são colocados em 0
    # Com isso são calculados automaticamente com base no tamanho do kernel
    blurred_image = cv2.GaussianBlur(
        gray_image, (kernel_size, kernel_size), 0
    )

    # Divide a imagem em escala de cinza pela imagem desfocada
    # para criar o efeito de desenho a lápis.
    # Reescacala para a escala de 256
    pencil_image = cv2.divide(gray_image, blurred_image, scale=256)

    return pencil_image


def brightness_adjustment(image: np.array, factor: float) -> np.array:
    """
    Ajusta o brilho da imagem dada pelo fator especificado.
    Parâmetros de entrada:
        - image: Imagem monocromática.
        - factor: O fator de ajuste de brilho.
    Saída da função:
        Imagem com brilho ajustado.
    OBS:
        Se o fator for menor ou igual a 0, retorna a imagem original.
        Fatores maiores que 1 aumentam o brilho
        Fatores entre 0 e 1 diminuem o brilho.
    """
    # Se o fator for menor ou igual a 0, retorna a imagem original
    if factor <= 0:
        return image

    # Converte as intensidades de [0,255] para [0,1]
    new_scale_image = image / 255

    # Ajusta o brilho
    adjusted_image = new_scale_image**(1/factor)

    # Re-escala novamente para [0,255] pela fórmula linear
    # (new_pixel_value - 0)/(255 - 0) = (old_pixel_value - 0)/(1 - 0)
    # new_pixel_value = old_pixel_value * 255
    # Trunca os valores dentro de [0, 255]
    # Converte para uint8 (tipo original dos pixels)
    # Neste caso a fórmula linear funciona por estar entre 0 e 1
    # de forma linear na escala anterior
    adjusted_image = np.clip((adjusted_image * 255), 0, 255).astype(np.uint8)

    return adjusted_image


def limiar_binarization(image: np.array, threshold: int) -> np.array:
    """
    Aplica a limiarização binária de uma imagem monocromática por um
    threshold.
    Parâmetros de entrada:
        - image: Imagem monocromática.
        - threshold: Threshold para a limiarização binária.
    Saída da função:
        Imagem transformada com a limiarização binária.
    """
    # Trunca o valor do threshold para o intervalo [0, 255]
    threshold = np.clip(threshold, 0, 255)

    # Aplica a limiarização binária
    # pixels com valor acima do limiar recebem o valor máximo (255)
    # picels com valor abaixo ou igual ao limiar recebem o valor mínimo (0)
    return np.where(image > threshold, 255, 0).astype(np.uint8)


def mosaic(
        image: np.array,
        matrix_shape: int = (4,4),
        map_matrix: np.array = None,
) -> np.array:
    """
    Aplica um efeito de mosaico em uma imagem monocromática,
    dividindo a imagem em blocos e reordenando eles.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - matrix_shape: A forma da matriz de blocos para o efeito de mosaico.
            Padrão é (4,4).
            Se map_matrix for fornecida, matrix_shape terá a dimensão dela.
        - map_matrix: Uma matriz que mapeia onde os índices dos blocos
            originais estarão.
            Ex: Se o valor de uma posição for 1, o primeiro bloco original
                estará nesta posição no mosaico. Por padrão a indexação dos
                blocos foi iniciada como 1.
            Se None, uma matriz aleatória é gerada
    Saída da função:
        Imagem com o efeito de mosaico.
    """
    # Define o número de divisões a serem feitas em cada dimensão
    n_blocks_x, n_blocks_y = matrix_shape[0], matrix_shape[1]

    # Se a matriz de mapeamento não é fornecida, gera uma matriz aleatória
    if map_matrix is None:
        # por padrão, a posição inicial é definida como 1 (por isso soma 1)
        map_matrix = np.random.permutation(
            n_blocks_x * n_blocks_y
        ).reshape(n_blocks_x, n_blocks_y) + 1
    # Se a matriz é fornecida, o número de divisões é definido a partir dela.
    else:
        n_blocks_x, n_blocks_y =  map_matrix.shape[0],  map_matrix.shape[1]

    # Gera um erro se o número de blocos não for um divisor
    #  das dimensões da imagem.
    if image.shape[0] % n_blocks_x != 0 or image.shape[1] % n_blocks_y != 0:
        raise ValueError("Imagem não é divisível pelo número de blocos.")

    # Calcula a altura e largura de cada bloco (divisões)
    block_height = image.shape[0] // n_blocks_x
    block_width = image.shape[1] // n_blocks_y

    # Separação da imagem em blocos na dimensão
    # (n_blocks_x, block_height, n_blocks_y, block_width)
    blocks = image.reshape(n_blocks_x, block_height, n_blocks_y, block_width)

    # Trasposição para o formato:
    # (n_blocks_x, n_blocks_y, block_height, block_width)
    # Esta transposição é necessária para facilitar a reordenação dos blocos.
    blocks = blocks.transpose(0, 2, 1, 3)

    # Junção das primeiras dimensões:
    # (n_blocks_x*n_blocks_y, block_height, block_width)
    # Obtém uma lista de blocos, indexada pela posição
    blocks_flattened = blocks.reshape(-1, block_height, block_width)

    # Reordena os blocos de acordo com os valores da map_matrix
    # (-1 para voltar para indexação padrão baseada em 0)
    reordered = blocks_flattened[map_matrix.flatten()-1]

    # Voltando para a dimensão matricial:
    # (n_blocks_x, n_blocks_y, block_height, block_width)
    reordered = reordered.reshape(
        n_blocks_x, n_blocks_y, block_height, block_width
    )

    # Restaurando a disposição original dos blocos:
    # (n_blocks_x, block_height, n_blocks_y, block_width)
    reordered = reordered.transpose(0, 2, 1, 3)

    # Restaurando a dimensão original da imagem:
    # (block_height*n_blocks_x, block_width*n_blocks_y)
    mosaic_image = reordered.reshape(
        block_height * n_blocks_x, block_width * n_blocks_y
    )

    return mosaic_image


def old_filter(image: np.array) -> np.array:
    """
    Aplica um filtro de imagem antiga a uma imagem colorida
    por operação matricial linear.
    Parâmetros de entrada:
        - image: Imagem colorida original em formato BGR (padrão do OpenCV).
    Saída da função: Imagem com o filtro de imagem antiga aplicado.
    """

    # Conversão da imagem para o formato RGB,
    # A matriz de transformação é baseada nesse formato
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Matriz de transformação para o filtro de imagem antiga
    trans_matrix = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])

    # Aplica a matriz de transformação linear
    img_old = image @ trans_matrix.T

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    img_old = np.clip(img_old, 0, 255).astype(np.uint8)

    # Conversão da imagem para o formato BGR, padrão do OpenCV
    img_old = cv2.cvtColor(img_old, cv2.COLOR_RGB2BGR)

    return img_old


def old_filter2(image: np.array) -> np.array:
    """
    Aplica um filtro de imagem antiga a uma imagem colorida por
    transformações individuais nos canais.
    Parâmetros de entrada:
        - image: Imagem colorida original em formato BGR (padrão do OpenCV).
    Saída da função: Imagem com o filtro de imagem antiga aplicado.
    """

    # Separação dos canais de cor da imagem (BGR)
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    # Aplica a matriz de transformação linear
    R_new = 0.393 * R + 0.769 * G + 0.189 * B
    G_new = 0.349 * R + 0.686 * G + 0.168 * B
    B_new = 0.272 * R + 0.534 * G + 0.131 * B

    # Junta novamente os canais em BGR
    img_old = np.stack([B_new, G_new, R_new], axis=2)

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    img_old = np.clip(img_old, 0, 255).astype(np.uint8)

    return img_old


def color2gray(image: np.array) -> np.array:
    """
    Converte uma multicanais para um canal.
    Parâmetros de entrada:
        - image: Imagem colorida original em formato BGR (padrão do OpenCV).
    Saída da função: Imagem com um único canal (aproxima a grayscale)
    """
    # Separação dos canais de cor da imagem (BGR)
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    # Converte 3 canais para um canal por meio de uma combinação linear
    new_image = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    return np.clip(new_image, 0, 255).astype(np.uint8)


def bit_plane(image: np.array, plane: int) -> np.array:
    """
    Extrai um plano de bits de uma imagem monocromática.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - plane: O número do plano de bits a ser extraído
            (0 para o bit menos significativo, 7 para o mais significativo).
    Saída da função: Imagem binária representando o plano de bits extraído.
    """
    # Trunca o número do plano para o intervalo [0, 7]
    plane = np.clip(plane, 0, 7)

    # Desloca os bits de cada pixel para a direita pelo número do plano
    # Operação AND com 1 binário para isolar o último bit
    # Obtém o bit referente ao plano de bits desejado
    bit_plane = (image >> plane) & 1

    # Imagem no plano de bits = bit*2^plano
    bit_plane_img = bit_plane * 2**plane

    # trunca os valores para o intervalo [0, 255] e converte para uint8
    return np.clip(bit_plane_img, 0, 255).astype(np.uint8)


def merge_images(
        image1: np.array,
        image2: np.array,
        ratio: float
) -> np.array:
    """
    Mescla duas imagens monocromáticas pela média ponderada dos pixels.
    Parâmetros de entrada:
        - image1: imagem monocromática.
        - image2: imagem monocromática.
        - ratio:  A proporção da primeira imagem na mesclagem
            (0 <= ratio <= 1).
    Saída da função: Imagem resultante que mescla as duas imagens.
    OBS:
        As imagens de entrada devem ter as mesmas dimensões.
    """
    # As imagens devem ter as mesmas dimensões
    if image1.shape != image2.shape:
        raise ValueError(
            "As imagens devem ter as mesmas dimensões para serem mescladas."
        )

    # Trunca o valor do ratio para o intervalo [0, 1]
    ratio = np.clip(ratio, 0, 1)

    # Valores dos pixels são a média ponderada dos pixels das duas imagens
    merged_image = ratio * image1 + (1 - ratio) * image2

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    return np.clip(merged_image, 0, 255).astype(np.uint8)


def _negative(image: np.array) -> np.array:
    """
    Aplica o efeito de negativo a uma imagem monocromática.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
    Saída da função: Imagem transformada com o efeito de negativo.
    """
    # O efeito de negativo é obtido subtraindo de 255 os valores dos pixels
    return np.clip(255 - image, 0, 255).astype(np.uint8)


def _rescale(image: np.array, interval: tuple) -> np.array:
    """
    Reescala as intensidades de uma imagem monocromática para um novo
    intervalo.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - interval: O novo intervalo para as intensidades (min, max).
    Saída da função: Imagem com as intensidades reescaladas para o novo
        intervalo.
    """

    # Converte para float para evitar overflow
    image = image.astype(np.float32)

    # Determina o mínimo e o máximo do intervalo original das intensidades
    original_min, original_max = image.min(), image.max()

    # Determina o novo intervalo das intensidades
    new_min, new_max = interval

    # Reescala as intensidades para o novo intervalo pela fórmula linear
    #(new_pixel_value - new_min)/(new_max - new_min) = (old_pixel_value - original_min)/(original_max - original_min)
    rescaled_image = (((image - original_min) * (new_max - new_min)) / (original_max - original_min)) + new_min

    # Trunca valores no intervalo [new_min, new_max] e converte para uint8
    return  np.clip(rescaled_image, new_min, new_max).astype(np.uint8)


def _rescale_cv(image: np.array, interval: tuple) -> np.array:
    """
    Reescala as intensidades de uma imagem monocromática para um novo
    intervalo usando funções prontas do OpenCV.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - interval: O novo intervalo para as intensidades (min, max).
    Saída da função: Imagem com as intensidades reescaladas para o novo
        intervalo.
    """
    # Reescala pela função de normalização minmax
    return cv2.normalize(
        image, None, interval[0], interval[1], cv2.NORM_MINMAX
    )


def _clip(image: np.array, interval: tuple) -> np.array:
    """
    Limita as intensidades de uma imagem monocromática a um
    intervalo específico.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - interval: O intervalo para limitar as intensidades (min, max).
    Saída da função: Imagem com as intensidades limitadas ao
        intervalo especificado.
    """
    # Trunca os valores para o intervalo especificado e converte para uint8
    return np.clip(image, interval[0], interval[1]).astype(np.uint8)


def _even_inverter(image: np.array) -> np.array:
    """
    Inverte as linhas pares de uma imagem monocromática.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
    Saída da função: Imagem com as linhas pares invertidas.
    """
    # Cria uma cópia da imagem para evitar modificar a imagem original
    # Passo necessário pois alterações são feitas na imagem
    inverted_image = image.copy()
    # Seleciona as linhas pares (índices 0, 2, 4, ...)
    # [:, ::-1] inverte a ordem dos pixels em cada linha
    inverted_image[::2] = inverted_image[::2][:, ::-1]

    return inverted_image


def _middle_mirror(image: np.array) -> np.array:
    """
    Espelha a metade superior da imagem
    Parâmetros de entrada:
        - image: Imagem monocromática original.
    Saída da função: A imagem espelhada horizontalmente no meio.
    """
    # Parte superior da imagem
    top = image[: len(image) // 2, :]
    # Concatena a metade superior com a metade superior com linhas invertidas
    return np.concatenate([top, top[::-1,:]], axis=0)


def _vertical_mirror(image: np.array) -> np.array:
    """
    Espelhamento vertical da imagem.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
    Saída da função: A imagem espelhada verticalmente no meio.
    """
    # Espelha a imagem invertendo a ordem das linhas
    return image[::-1,:]


def intensity_transformation(
        image: np.array,
        function: callable,
        **kwargs
) -> np.array:
    """
    Aplica uma transformação de intensidade a uma imagem monocromática.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - function: Função que define a transformação de intensidade
            a ser aplicada. (Ex _middle_mirror, _negative, etc)
    Saída da função: Imagem resultante da aplicação da transformação de
        intensidade.
    """
    # Aplica a função de transformação de intensidade à imagem
    return function(image, **kwargs)


def quantization(image: np.array, n_levels: int) -> np.array:
    """
    Quantiza uma imagem monocromática para um número específico de
    níveis de intensidade.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - n_levels: O número de níveis de intensidade desejados.
    Saída da função: Imagem quantizada com o número especificado de
        níveis de intensidade.
    OBS:
        O número de níveis deve ser um divisor exato do intervalo total
        de intensidades (256 para imagens de 8 bits).
    """
    # O número de níveis deve ser um divisor de 256 para imagens de 8 bits
    if 256 % n_levels != 0:
        raise ValueError(
            "O número de níveis deve ser um divisor exato de 256."
        )

    # Passo de quantização = 256 / n_levels -> Intervalo de cada nível
    step = 256 // n_levels

    # Para obter o novo valor de cada pixel basta:
    # Dividir de forma inteira o valor pelo passo de quantização
    # Obtendo, assim, o índice do nível correspondente
    # Muliplicar este valor de índice pelo passo,
    # Para obter o valor de intensidade para este índice
    quantized_image = (image // step) * step

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    return np.clip(quantized_image, 0, 255).astype(np.uint8)


def _opencv_filtering(
        image: np.array,
        kernel: np.array,
        borderType: Any = cv2.BORDER_REFLECT_101
    ) -> np.array:
    """
    Aplica um filtro a uma imagem monocromática por meio de convolução.
    Usa o Opencv (otimizada)
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - kernel: O filtro a ser aplicado (matriz).
        - borderType: O tipo de borda a ser usado durante a convolução.
            cv2.BORDER_REFLECT_101 (padrão):
            Reflete a borda da imagem, excluindo os pixels da borda.
    Saída da função: Imagem resultante da aplicação do filtro à imagem
        original.
    """
    # Passa a imagem para float32, evitando overflow numérico
    image = image.astype(np.float32)
    # Aplica o filtro à imagem usando a função de convolução do OpenCV
    return cv2.filter2D(image, -1, kernel, borderType=borderType)


def border_reflect_101(
        image: np.array,
        border_size_h: int,
        border_size_w: int
    ) -> np.array:
    """
    Aplica o tipo de borda BORDER_REFLECT_101 a uma imagem monocromática sem
    utilizar o opencv
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - border_size_h: Tamanho da borda a ser aplicado na altura da imagem.
        - border_size_w: Tamanho da borda a ser aplicado na largura da imagem.
    Saída da função: Imagem com a borda aplicada.
    """
    # Aplica a borda usando a função np.pad com o modo 'reflect' do numpy,
    return  np.pad(
        image,
        ((border_size_h, border_size_h), (border_size_w, border_size_w)),
        mode='reflect'
    )


def _numpy_filtering(
        image: np.array,
        kernel: np.array,
        borderType: Any = border_reflect_101
    ) -> np.array:
    """
    Aplica um filtro a uma imagem monocromática por meio de convolução.
    Implementação com numpy vetorizada
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - kernel: O filtro a ser aplicado (matriz).
        - borderType: O tipo de borda a ser usado durante a convolução.
            border_reflect_101 (padrão):
            Reflete a borda da imagem, excluindo os pixels da borda.
    Saída da função: Imagem resultante da aplicação do filtro à imagem
        original.
    """
    # Calcula o acréscimo de borda necessário pelas dimensões do filtro
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Aplica a adição de borda
    padded_image = borderType(image, pad_h, pad_w)

    # Aplicação de janelas desliantes vetorizadas
    # Extrai o shape das janelas deslizantes (H, W, kh, kw)
    # Imagem (H, W), filtro (kh, kw)
    # Janelas (H, W, kh, kw)
    windows_shape  = (
        image.shape[0],
        image.shape[1],
        kh,
        kw
    )

    # Passos a serem dados nas dimensões para obter as janelas deslizantes
    strides = (
        padded_image.strides[0], # mover em H
        padded_image.strides[1], # mover em W
        padded_image.strides[0], # mover em kh
        padded_image.strides[1] # mover em kw
    )

    # Criação das janelas deslizantes usando a função as_strided do numpy
    sliding_windows = np.lib.stride_tricks.as_strided(
        padded_image,
        shape=windows_shape,
        strides=strides
    )

    # Operação de convolução vetorizada usando a função einsum do numpy
    # (H, W, kh, kw) = (i, j, k, l) e (kh, kw) = (k, l) -> resultado é (H, W) = (i, j)
    # ijkl: sliding_windows, kl: kernel -> ij: resultado da convolução
    filtered_image = np.einsum('ijkl,kl->ij', sliding_windows, kernel)

    return filtered_image


def _loop_filtering(
        image: np.array,
        kernel: np.array,
        borderType: Any = border_reflect_101
    ) -> np.array:
    """
    Aplica um filtro a uma imagem monocromática por meio de convolução.
    Usando loops aninhados (forma mais simples)
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - kernel: O filtro a ser aplicado (matriz).
        - borderType: O tipo de borda a ser usado durante a convolução.
            border_reflect_101 (padrão):
            Reflete a borda da imagem, excluindo os pixels da borda.
    Saída da função:/Imagem resultante da aplicação do filtro à imagem
        original.
    """
    # Calcula o acréscimo de borda necessário
    kh, kw = kernel.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Imagem após a adição da borda
    padded_image = borderType(image, pad_h, pad_w)

    # Inicialização da imagem resultante da convolução
    H, W = image.shape
    filtered_image = np.zeros((H, W), dtype=np.float32)

    # Operação de convolução
    for i in range(H):
        for j in range(W):
            acc = 0.0
            # Para cada pixel, percorre o kernel (kh, kw) = (k, l)
            # calcula a soma dos produtos entre os pixels e kernels
            for k in range(kh):
                for l in range(kw):
                    acc += (
                        padded_image[i + k, j + l] *
                        kernel[k, l]
                    )
            # Salva o valor calculado para o pixel (i, j) da imagem resultante
            filtered_image[i, j] = acc

    return filtered_image


def filtering(
        image: np.array,
        kernel: np.array,
        borderType: callable = cv2.BORDER_REFLECT_101,
        function: callable = _opencv_filtering
    ) -> np.array:
    """
    Aplica um filtro a uma imagem monocromática por meio de convolução.
    Através de uma função definida
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - kernel: O filtro a ser aplicado (matriz).
        - borderType: O tipo de borda a ser usado durante a convolução.
            border_reflect_101 (padrão)
            Reflete a borda da imagem, excluindo os pixels da borda.
        - function: A função de filtragem a ser aplicada.
            Padrão é a função de filtragem do opencv (_opencv_filtering)
    Saída da função: Imagem resultante da aplicação do filtro à imagem
        original.
    OBS:
        O kernel deve ter dimensões ímpares para garantir um centro
        definido na convolução.
    """
    # O kernel deve ter dimensões ímpares
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("O kernel deve ter dimensões ímpares.")

    # Aplica o filtro e mede o tempo de duração da operação
    start = time.perf_counter()
    filtered_image = function(image, kernel, borderType)
    end = time.perf_counter()

    # Mostra os tempo de execição
    print(f"{function.__name__}: {end - start:.4f}.")

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    return np.clip(filtered_image, 0, 255).astype(np.uint8)


def combining_filters(
        image: np.array,
        kernel1: np.array,
        kernel2: np.array,
        borderType: callable = cv2.BORDER_REFLECT_101,
        function: callable = _opencv_filtering
    ) -> np.array:
    """
    Aplica uma combinação de filtros a uma imagem monocromática
    por meio de convolução.
    Parâmetros de entrada:
        - image: Imagem monocromática original.
        - kernel1: O primeiro kernel do filtro a ser aplicado.
        - kernel2: O segundo kernel do filtro a ser aplicado.
         - borderType: O tipo de borda a ser usado durante a convolução.
            border_reflect_101 (padrão)
            Reflete a borda da imagem, excluindo os pixels da borda.
        - function: A função de filtragem a ser aplicada.
            Padrão é a função de filtragem do opencv (_opencv_filtering)
    Saída da função:
        Imagem resultante da aplicação da combinação de filtros à
        imagem original.
    OBS:
        O kernel deve ter dimensões ímpares para garantir um centro
        definido na convolução.
    """
    # O kernel deve ter dimensões ímpares
    for kernel in [kernel1, kernel2]:
        if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
            raise ValueError("Os kernels devem ter dimensões ímpares.")

    # Mede o tempo de execução
    start = time.perf_counter()
    # Cria uma cópia da imagem original
    original_image = image.copy()
    # Resposta da aplicação do primeiro filtro
    filtered_image1 = function(original_image, kernel1, borderType)
    original_image = image.copy()
    # Resposta da aplicação do segundo filtro
    filtered_image2 = function(original_image, kernel2, borderType)

    # combinação das respostas dos filtros
    filtered_image = np.sqrt(filtered_image1**2 + filtered_image2**2)
    end = time.perf_counter()

    # Mostra o tempo de execução
    print(f"{function.__name__}: {end - start:.4f}.")

    # Trunca os valores para o intervalo [0, 255] e converte para uint8
    return np.clip(filtered_image, 0, 255).astype(np.uint8)
