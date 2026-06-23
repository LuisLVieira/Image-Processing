"""
Realiza transformações geométricas em imagens coloridas e monocromáticas
para rotação e translação

Execução

transformações_geométricas.py [-a ângulo]
[-e fator de escala]
[-d largura altura]
[-m interpolação]
[-i imagem]
[-o imagem]
em que os parâmetros são:
-a ângulo de rotação medido em graus no sentido anti-horário
-e fator de escala
-d dimensão da imagem de saída em pixels
-m método de interpolação utilizado (near_neighbor, bilinear, bicubic or lagrange)
-i imagem de entrada no formato PNG
-o imagem de saída no formato PNG (após a transformação geométrica)
"""



import os
import cv2
import numpy as np
import argparse
from typing import Tuple
import time


def read_args():
    """
    Decodifica os argumentos passados na execução do programa.
    Parâmetros de entrada:
        -a ângulo de rotação medido em graus no sentido anti-horário (float)
        -e fator de escala (float)
        -d dimensão da imagem de saída em pixels (str str)
        -m método de interpolação utilizado (string) (near_neighbor, bilinear, bicubic or lagrange)
        -i imagem de entrada no formato PNG (string)
        -o imagem de saída no formato PNG (após a transformação geométrica) (string)
    Saídas da função:
        - args: argumentos de entrada
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        type=float,
        default=None,
        help="ângulo de rotação medido em graus no sentido anti-horário"
    )
    parser.add_argument(
        "-e",
        type=float,
        default=None,
        help="fator de escala"
    )
    parser.add_argument(
        "-d",
        nargs=2,
        default=None,
        metavar=("LARGURA", "ALTURA"),
        help="dimensões da imagem de saída (largura altura)"
    )
    parser.add_argument(
        "-m",
        type=str,
        default=None,
        help="método de interpolação utilizado"
    )
    parser.add_argument(
        "-i",
        type=str,
        default=None,
        help="imagem de entrada"
    )
    parser.add_argument(
        "-o",
        type=str,
        default=None,
        help="imagem de saída"
    )

    return parser.parse_args()


def scale(xo: np.array, yo: np.array, scale_factor: float) -> Tuple[np.array]:
    """
    Aplica o mapeamento inverso de escala, retornando as coordenadas
    da imagem de entrada referentes às coordenadas de saída
    Parâmetros de entrada:
        -xo: coordenadas x da imagem de saída
        -yo: coordenadas y da imagem de saída
        -scale_factor: fator de escala da transformação

    Saídas da função:
        - xi: respectivas coordenadas x da imagem de entrada
        - yi: respectivas coordenadas y da imagem de entrada
    """

    xi =  xo / abs(scale_factor)
    yi =  yo / abs(scale_factor)

    return xi, yi


def translation_matrix(tx: float, ty: float) -> np.array:
    """
    Retorna a matriz de translação dados o valor de translação em x e y
    Parâmetros de entrada:
        -tx: deslocamento em x
        -ty: deslocamento em y

    Saídas da função:
        Matriz de translação
    """

    return np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ])


def rotation(
    xo: np.array,
    yo: np.array,
    angle: float,
    input_dim: tuple,
    output_dim: tuple
) -> Tuple[np.array]:
    """
    Aplica o mapeamento inverso de rotação, retornando as coordenadas
    da imagem de entrada referentes às coordenadas de saída
    Parâmetros de entrada:
        -xo: coordenadas x da imagem de saída
        -yo: coordenadas y da imagem de saída
        -angle: ângulo de rotação em graus
        -input_dim: dimensão da imagem de entrada (largura, altura)
        -output_dim: dimensão da imagem de saída (largura, altura)

    Saídas da função:
        - xi: respectivas coordenadas x da imagem de entrada
        - yi: respectivas coordenadas y da imagem de entrada
    """

    # ângulo em radianos
    theta = np.deg2rad(angle)

    w_in, h_in = input_dim
    w_o, h_o = output_dim

    # pontos centrais das imagens
    cx_in = w_in / 2.0
    cy_in = h_in / 2.0
    cx_out = w_o / 2.0
    cy_out = h_o / 2.0

    # Matriz de translação para o ponto central
    # Obtém coordenadas relativas ao centro para
    # a rotação ser feita com relação ao centro e
    # não a origem (borda superior esquerda)

    # O centro da imagem de saída se torna a posição (0,0)
    # Move o centro da imagem de saída para a origem
    T_out = translation_matrix(-cx_out, -cy_out)


    # matriz de rotação com relação à origem
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])


    # Desloca positivamente a imagem de entrada para retornar às coordenadas originais
    # Move a origem da imagem de entrada para o centro
    T_in = translation_matrix(cx_in, cy_in)

    # Matriz de rotação com relação ao centro
    M = T_in @ R @ T_out

    # Coordenadas homogênias da saída
    coord_out = np.stack(
        (
            xo.ravel(),
            yo.ravel(),
            np.ones(xo.size)
        ),
        axis=0
    )

    # Coordenadas homogêneas de entrada
    coord_in = M @ coord_out

    # retornando para o sistema de coordenadas x, y na imagem de entrada
    xi = coord_in[0].reshape(xo.shape)
    yi = coord_in[1].reshape(yo.shape)

    return xi, yi


def get_pixel_value(img: np.array, x: np.array, y: np.array) -> np.array:
    """
    Retorna os valores dos pixels da imagem nas coordenadas especificadas.

    Dadas coordenadas (x,y) de uma imagem, retorna os valores dos pixels.
    Para casos de posições fora da dimensão da imagem, o valor do pixel foi
    definido como a cópia do valor da borda, possibilitando o cálculo das
    funções de interpolação em pixels de borda para vizinhos inexistentes. 
    Parâmetros de entrada:
        -img: imagem
        - x: coordenadas x dos pixels a serem acessados.
        - y: coordenadas y dos pixels a serem acessados.

    Saídas da função:
        - value: valores dos pixels correspondentes às coordenadas fornecidas,
          retornados em formato float64. (N, 1) em imagems monocromáticas.
          e (N, C) em imagens coloridas, sendo C os canais.
    """
    h, w = img.shape[:2]

    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    value = img[y, x].astype(np.float64)

    # se caso for monocromática, adiciona mais uma dimensão
    if value.ndim == 1:
        value = value.reshape(-1, 1)

    return value


def near_neighbor(img: np.array, xf: np.array, yf: np.array) -> np.array:
    """
    Aplica a interpolação por vizinho mais próximo de maneira vetorizada.
    Arredonda cada valor de coordenada obtidos nas transformações, para obter
    os pixels vizinhos. O valor do pixel será o valor do do vizinho mais próximo
    (menor distância do valor da coordenada).
    Parâmetros de entrada:
        -img: imagem
        -x: valores da coordenada x após a transformação.
        -y: valores da coordenada y após a transformação.

    Saídas da função:
        - valor do pixel
    """
    # Arredonadamento dos valores após a transformação
    # Obtenção dos vizinhos
    x = np.floor(xf).astype(int)
    y = np.floor(yf).astype(int)

    # distâncias para os vizinhos
    dx = xf - x
    dy = yf - y

    # coordenadas do vizinho mais próximo
    xn = np.where(dx < 0.5, x, x + 1)
    yn = np.where(dy < 0.5, y, y + 1)

    # Retorna a imagem interpolada
    return get_pixel_value(img, xn, yn)


def bilinear(img: np.array, xf: np.array, yf: np.array) -> np.array:
    """
    Aplica a interpolação bilinear de maneira vetorizada.
    Arredonda cada valor de coordenada obtidos nas transformações, para obter
    os pixels vizinhos. O valor do pixel será uma combinação ponderada dos
    pixels vizinhos.
    Parâmetros de entrada:
        -img: imagem
        -x: valores da coordenada x após a transformação.
        -y: valores da coordenada y após a transformação.

    Saídas da função:
        - valor do pixel
    """

    # Arredonadamento dos valores após a transformação
    # Obtenção dos vizinhos
    x = np.floor(xf).astype(np.int32)
    y = np.floor(yf).astype(np.int32)

    # distâncias para os vizinhos
    dx = (xf - x).reshape(-1, 1)
    dy = (yf - y).reshape(-1, 1)

    # valores dos 4 pixels vizinhos
    f1 = get_pixel_value(img, x, y)
    f2 = get_pixel_value(img, x + 1, y)
    f3 = get_pixel_value(img, x, y + 1)
    f4 = get_pixel_value(img, x + 1, y + 1)

    # Retorna a imagem interpolada
    return ((1 - dx) * (1 - dy) * f1 +
            dx * (1 - dy) * f2 +
            (1 - dx) * dy * f3 +
            dx * dy * f4)


def _P(t: np.array) -> np.array:
    """
    Aplica uma função em um array em que os valores negativos
    se tornam zero. Auxiliar para bicubic.
    Parâmetros de entrada:
        -t: array de entrada

    Saídas da função:
        - array com a função aplicada.
    """
    return np.maximum(t, 0)


def _R(s: np.array) -> np.array:
    """
    Aplica a função em um array para obter valores positivos
    de pesos por uma função cúbica. Auxiliar para bicubic.
    Parâmetros de entrada:
        -t: array de entrada

    Saídas da função:
        - array com a função aplicada.
    """
    return (1.0 / 6.0) * (
        _P(s + 2) ** 3
        - 4 * _P(s + 1) ** 3
        + 6 * _P(s) ** 3
        - 4 * _P(s - 1) ** 3
    )


def bicubic(img: np.array, xf: np.array, yf: np.array) -> np.array:
    """
    Aplica a interpolação bicubica de maneira vetorizada.
    Arredonda cada valor de coordenada obtidos nas transformações, para obter
    os pixels vizinhos. O valor do pixel será uma combinação dos
    pixels vizinhos pela função B-spline cúbica. Obtém os pesos cúbicos
    pela função _R.
    Parâmetros de entrada:
        -img: imagem
        -x: valores da coordenada x após a transformação.
        -y: valores da coordenada y após a transformação.

    Saídas da função:
        - valor do pixel
    """

    # Arredonadamento dos valores após a transformação
    # Obtenção dos vizinhos
    x = np.floor(xf).astype(np.int32)
    y = np.floor(yf).astype(np.int32)

    # distâncias para os vizinhos
    dx = (xf - x).reshape(-1, 1)
    dy = (yf - y).reshape(-1, 1)

    result = None
    # w are the weights
    for m in range(-1, 3):
        w_m = _R(m - dx)
        for n in range(-1, 3):
            w = w_m * _R(dy - n)
            value = get_pixel_value(img, x + m, y + n)
            result = value * w if result is None else result  + value * w

    return result

def _L(img, x, y, dx, n):
    """
    Cálculo do polinômio de lagrange
    Parâmetros de entrada:
        -img: imagem
        -x: valores da coordenada x após a transformação.
        -y: valores da coordenada y após a transformação.
        -dx: distância ao vizinho na coordenada x
        -n: ordem

    Saídas da função:
        - valor do pixel
    """
    f1 = get_pixel_value(img, x - 1, y + n - 2)
    f2  = get_pixel_value(img, x,     y + n - 2)
    f3 = get_pixel_value(img, x + 1, y + n - 2)
    f4 = get_pixel_value(img, x + 2, y + n - 2)

    termo1 = (-dx * (dx - 1) * (dx - 2) / 6.0) * f1
    termo2 = ((dx + 1) * (dx - 1) * (dx - 2) / 2.0) * f2
    termo3 = (-dx * (dx + 1) * (dx - 2) / 2.0) * f3
    termo4 = (dx * (dx + 1) * (dx - 1) / 6.0) * f4

    return termo1 + termo2 + termo3 + termo4



def lagrange(img: np.array, xf: np.array, yf: np.array) -> np.array:
    """
    Aplica a interpolação por polinômios de lagrange de maneira vetorizada.
    Arredonda cada valor de coordenada obtidos nas transformações, para obter
    os pixels vizinhos. O valor do pixel será uma combinação dos
    pixels vizinhos pela função polinômio de lagrange.
    Parâmetros de entrada:
        -img: imagem
        -x: valores da coordenada x após a transformação.
        -y: valores da coordenada y após a transformação.

    Saídas da função:
        - valor do pixel
    """
    # Arredonadamento dos valores após a transformação
    # Obtenção dos vizinhos
    x = np.floor(xf).astype(np.int32)
    y = np.floor(yf).astype(np.int32)

    # distâncias para os vizinhos
    dx = (xf - x).reshape(-1, 1)
    dy = (yf - y).reshape(-1, 1)

    L1 = _L(img, x, y, dx, 1)
    L2 = _L(img, x, y, dx, 2)
    L3 = _L(img, x, y, dx, 3)
    L4 = _L(img, x, y, dx, 4)

    termo1 = (-dy * (dy - 1) * (dy - 2) / 6.0) * L1
    termo2 = ((dy + 1) * (dy - 1) * (dy - 2) / 2.0) * L2
    termo3 = (-dy * (dy + 1) * (dy - 2) / 2.0) * L3
    termo4 = (dy * (dy + 1) * (dy - 1) / 6.0) * L4

    return termo1 + termo2 + termo3 + termo4

interpolations = {
    "near_neighbor": near_neighbor,
    "bilinear": bilinear,
    "bicubic": bicubic,
    "lagrange": lagrange
}


def apply_transform(
    img: np.array,
    interpolation: str,
    output_size: tuple | None=None,
    scale_factor: float | None=None,
    rotate_angle: float | None=None,
):
    """
    Aplica transformações em imagens de entrada. Como especificado no
    enunciado, apenas uma transformação (rotação ou escala) podem ser
    realizadas em uma única execução.
    Parâmetros de entrada:
        -img: imagem de entrada
        -interpolation: função de interpolação dos valores dos pixels
          Podendo ser:
            - near_neighbor
            - bilinear
            - bicubic
            - lagrange
        -output_size: dimensões da imagem de entrada. Se None a função obtém uma
        dimensão automática.
        -scale_factor: fator de escala
        -rotate_angle: ângulo de rotação em graus

    Saídas da função:
        - img_out: imagem de saída transformada
    """
    if (
        (scale_factor is None and rotate_angle is None) or
        (scale_factor is not None and rotate_angle is not None)
    ):
        print("Warning: Pass a valid scale_fator value OR a valid rotate_factor value. Returning input image.")
        return img

    # Dimensões da imagem de entrada
    h_in, w_in = img.shape[0], img.shape[1]
    ch = img.shape[2]

    if output_size is None:
        output_size = (w_in, h_in)

    # Se output size for max, calcula automaticamente a dimensão após resize
    # No rotation permanece sendo a dimensão de entrada
    if (
            output_size[0] == "max" and
            output_size[1] == "max"
        ):
            if scale_factor is not None:
                w_o = max(1, int(round(w_in * scale_factor)))
                h_o = max(1, int(round(h_in * scale_factor)))

            elif rotate_angle is not None:
                w_o = w_in
                h_o = h_in
    else:
        w_o, h_o = output_size

    output_size = (w_o, h_o)

    w_o, h_o = tuple(map(int, output_size))
    # Inicia a imagem de saída apenas com pixels pretos (0)
    img_out = np.zeros((h_o, w_o, ch), dtype=np.float64)

    # coordenadas de saída
    xo, yo = np.meshgrid(
        np.arange(w_o),
        np.arange(h_o)
    )

    # Obtendo as coordenadas de entrada referentes a cada pixel
    # da imagem de saída após a transformação
    if scale_factor is not None:
        xi, yi = scale(xo, yo, scale_factor)

    elif rotate_angle is not None:
        xi, yi = rotation(
            xo,
            yo,
            rotate_angle,
            input_dim=(w_in, h_in),
            output_dim=(w_o, h_o)
        )

    # Máscara de pixels válidos
    # Mantém pixels fora da imagem como 0 (preto)
    # -1 seria a posição mínima válida para poder interpolar valores de borda,
    # que não tem todos os vizinhos existentes, nesse caso é considerada a 
    # replicação de borda dimensão + 1 é o máximo válido com replicação de borda.
    valid = (
        (xi >= -1) &
        (xi <= w_in + 1) &
        (yi >= -1) &
        (yi <= h_in + 1)
    )

    # Interpolação dos valores das coordenadas apenas em pontos válidos
    out_values = interpolations[interpolation](
        img,
        xi[valid],
        yi[valid]
    )

    # Obtendo a imagem de saída
    img_out[valid] = out_values

    # retorna a imagem de saída em uint8 e truncada de 0 a 255
    return np.clip(img_out, 0, 255).astype(np.uint8)


def main():
    start = time.time()
    args = read_args()
    if args.i is not None:
        img = cv2.imread(args.i)
        transformed_image = apply_transform(
            img,
            interpolation=args.m,
            output_size=args.d,
            scale_factor=args.e,
            rotate_angle=args.a
        )
        if args.o is not None:
            cv2.imwrite(
                args.o,
                transformed_image
            )
    end = time.time()
    print(f'processing time: {end - start} s')


if __name__ == "__main__":
    main()
