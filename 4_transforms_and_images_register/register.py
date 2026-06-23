"""
Realiza o registro de imagens, produzindo uma imagem panorâmica

Execução

registro.py [-i1 caminho da imagem 1]
[-i2 caminho da imagem 2]
[-o caminho (diretório) para salvar a imagem panorâmica]
[-d descritor de pontos de interesse]
Ex SIFT (Scale Invariant Feature Transform),
BRIEF (Robust Independent Elementary
Features),
ORB (Oriented FAST, Rotated BRIEF).
[-l limiar para considerar correspondência]
"""


import os
import cv2
import numpy as np
import argparse
from typing import Tuple, List
from scipy.spatial.distance import cdist
import time
import glob


descriptors = {
    "SIFT": cv2.SIFT_create,
    "ORB": cv2.ORB_create,
    "BRIEF": cv2.xfeatures2d.BriefDescriptorExtractor_create,
}



def read_args():
    """
    Decodifica os argumentos passados na execução do programa.
    Parâmetros de entrada:
        -i1 caminho da imagem 1 (string, obrigatório)
        -i2 caminho da imagem 2 (string, obrigatório)
        -o  diretório para salvar a imagem panorâmica (string, default '.')
        -d  descritor de pontos de interesse: SIFT (default), SURF, BRIEF, ORB
        -l  limiar para considerar correspondência (float, default 0.75)
    Saídas da função:
        - args: argumentos de entrada
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i1",
        type=str,
        required=True,
        help="caminho da imagem 1"
    )
    parser.add_argument(
        "-i2",
        type=str,
        required=True,
        help="caminho da imagem 2"
    )
    parser.add_argument(
        "-o",
        type=str,
        default=".",
        help="diretório onde salvar a imagem panorâmica e as correspondências"
    )
    parser.add_argument(
        "-d",
        type=str,
        default="SIFT",
        choices=list(descriptors.keys()),
        help="Descritor do ponto de interesse"
    )
    parser.add_argument(
        "-l",
        type=float,
        default=0.5,
        help="limiar para considerar correspondência "
    )

    return parser.parse_args()


def detect_and_descript(img: np.array, descriptor_name: str) -> Tuple[np.array]:
    """
    Detecta pontos de interesse e extrai descritores locais.

    Possíveis descritores:
        - SIFT
        - ORB
        - FAST + BRIEF

    Parâmetros de entrada:
        -img: imagem em escala de cinza
        -descriptor_name: Nome do descritor utilizado.
    Saídas da função:
        - kps : Lista de pontos de interesse detectados.
        - descs : Vetor de descritores associado aos pontos de interesse.
    """

    descriptor = descriptors[descriptor_name.upper()]()

    # O BRIEF é só um descritor, precisa um detector como o FAST
    if descriptor_name == "BRIEF":
        detector = cv2.FastFeatureDetector_create()
        kps = detector.detect(img, None)
        kps, descs = descriptor.compute(img, kps)
    else:
        kps, descs = descriptor.detectAndCompute(img, None)

    print(f'Foram encontrados {len(kps)} pontos de interesse')

    return kps, descs


def compute_descriptor_similarity(descs1: np.array, descs2: np.array) -> List:
    """
    Calcula a similaridade entre dois descritores por meio da Correlação Cruzada
    Normalizada.

    Parâmetros de entrada:
        -descs1: Descritores da primeira imagem.
        -descs2: Descritores da segunda imagem.
    Saídas da função:
        -pairs : Lista de pares de correspondências obtidas por meio de k-NN.
    """

    d1 = descs1.astype(np.float64)
    d2 = descs2.astype(np.float64)

    # Aplica a fórmula da Correlação
    # Subtrai a média de cada descritor para centralizar em zero
    d1c = d1 - d1.mean(axis=1, keepdims=True)
    d2c = d2 - d2.mean(axis=1, keepdims=True)

    # Cálcula a norma de cada descritor
    n1 = np.linalg.norm(d1c, axis=1, keepdims=True)
    n2 = np.linalg.norm(d2c, axis=1, keepdims=True)

    # Normaliza os descritores
    d1n = np.divide(d1c, n1, where=n1 > 0, out=np.zeros_like(d1c))
    d2n = np.divide(d2c, n2, where=n2 > 0, out=np.zeros_like(d2c))

    # Coeficientes de Correlação para todos os pares de descritores
    corr = d1n @ d2n.T

    # Melhores correlações
    best_idx = np.argmax(corr, axis=1)

    # Salva para cada descritor d1 sua maior correlação com d2
    matches = []
    for i in range(len(d1)):
        j = int(best_idx[i])
        matches.append(cv2.DMatch(i, j, float(1.0 - corr[i, j])))

    return matches, corr



def get_best_pairs(matches: list, corr: np.array, limiar: float) -> List:
    """
    Seleciona as correspondências cuja correlação de Pearson é maior ou
    igual ao limiar definido pelo usuário.


    Parâmetros de entrada:
        -matches: Lista de matches.
        -corr: Matriz de correlação.
        -limiar : Correlação mínima para aceitar a correspondência (0 a 1).
    Saídas da função:
        -best : Correspondências consideradas válidas (acima do limiar).
    """

    # le a lista de melhores correspondências e filtra as maiores que o limiar
    # para separar as válidas
    best = []
    for m in matches:
        if corr[m.queryIdx, m.trainIdx] >= limiar:
            best.append(m)
    return best


def estimate_homography(kps1: list, kps2: list, best_pairs: list) -> Tuple:
    """
    Estima a matriz de homografia entre as imagens. Utiliza o algoritmo
    RANSAC (RANdom SAmple Consensus)

    Parâmetros de entrada:
        -kps1: Pontos de interesse da imagem 1.
        -kps2: Pontos de interesse da imagem 2.
        -best_pairs: melhores correspondências
    Saídas da função:
        -H : Matriz de homografia 3x3.
        -mask: Máscara indicando as correspondências encontrados pelo RANSAC.
    """

    # Erro para o caso com menos de 4 correspondências (homografia exige 4)
    if len(best_pairs) < 4:
        raise ValueError(
            f"{len(best_pairs)} correspondêncis encontradas"
            "o cálculo da homografia exige no mínimo 4."
        )

    # Salva os pontos de interesse que possuem as correspondências válidas
    pts1 = np.float32([kps1[m.queryIdx].pt for m in best_pairs])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in best_pairs])

    # Cálculo da matriz de homografia e da máscara binária com as correspondências
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4)

    print('Matriz de homografia')
    print(H)
    return H, mask


def get_perspective(img1: np.array, img2: np.array, H: np.array):
    """
    Aplica a transformação projetiva de perspectiva na imagem 1.
    Calcula automaticamente a dimensão da image de saída para evitar
    cortes.

    Parâmetros de entrada:
        -img1: imagem 1.
        -img2: imagem 2.
        -H: matriz de homografia
    Saídas da função:
        -img1_perspective: Imagem 1 transformada.
        -dx: deslocamento em x
        -dy: deslocamento em y
    """

    # Dimensões das imagens
    h_1, w_1 = img1.shape[:2]
    h_2, w_2 = img2.shape[:2]

    # definição dos 4 vértices da imagem 1
    corners_1 = np.float32([[0, 0], [0, h_1], [w_1, h_1], [w_1, 0]]).reshape(-1, 1, 2)

    # Aplicação da transformação de perspectiva com a matriz de homografia
    # Para descobrir qual será a dimensão de saída
    proj_corners_1 = cv2.perspectiveTransform(corners_1, H)

    # definição dos 4 vértices da imagem 2
    corners_2 = np.float32([[0, 0], [0, h_2], [w_2, h_2], [w_2, 0]]).reshape(-1, 1, 2)

    # Obtém os novos cantos após a junção dos cantos após transformar
    corners = np.concatenate((proj_corners_1, corners_2), axis=0)

    # Novos valores inteiros para os limites
    x_min, y_min = np.floor(corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(corners.max(axis=0).ravel()).astype(int)

    # Matriz de translação para deslocar valores negativos de limites para 0
    T = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1],
    ], dtype=np.float64)

    # Tamanho calculado para a imagem de saída
    output_shape = (x_max - x_min, y_max - y_min)

    # Aplicação da transformação de perspectiva
    img1_perspective = cv2.warpPerspective(img1, T @ H, output_shape)

    return img1_perspective, -x_min, -y_min


def get_panoram(img2: np.array, img1_perspective: np.array, dx: int, dy:int) -> np.array:
    """
    Combina a segunda imagem com a imagem 1 transformada com a imagem 2 para
    montar o panorama.

    Parâmetros de entrada:
        -img2: imagem 2.
        -img1_perspective:  Imagem 1 transformada..
        -dx: deslocamento em x
        -dy: deslocamento em y
    Saídas da função:
        -panoramic: Imagem panorâmica final.
    """

    # dimensões da imagem 2
    h_2, w_2 = img2.shape[:2]

    # região onde a imagem 2 será inserida
    # Considera o deslocamento feito em get perspective
    region = img1_perspective[dy:dy + h_2, dx:dx + w_2]

    # máscara dos valores válidos da imagem 2
    # Elimina possíveis bordas com valor 0
    mask_2 = np.any(img2 > 0, axis=2)

    # insere os pixels válidos da imagem 2 na região correta
    region[mask_2] = img2[mask_2]

    # A operação anterior já modifica img1_perspective
    # Criei uma cópia em outra variável
    panoramic = img1_perspective.copy()

    return panoramic


def draw_correspondencies(
    img1: np.array,
    img2: np.array,
    kps1: list,
    kps2: list,
    best_pairs: list,
    mask: np.array
):
    """
    Desenha as correspondências válidas entre as imagens.

    Parâmetros de entrada:
        -img1 : Primeira imagem.
        -img2 : Segunda imagem.
        -kps1 : Pontos de interesse da imagem 1.
        -kps2 : Pontos de interesse da imagem 2.
        -best_pairs : Correspondências válidas.
        -mask : Máscara de correspondências do RANSAC.
    Saídas da função:
        -cors: Imagem contendo as correspondências desenhadas.
    """
    # Transforma a máscara de corresponência em uma lista
    matches_mask = mask.ravel().tolist()

    # Desenha as ligações dos pares
    cors = cv2.drawMatches(
        img1, kps1, img2, kps2, best_pairs, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return cors


def registry(
    img1: np.ndarray,
    img2: np.ndarray,
    limiar: float = 0.75,
    descriptor: str = 'SIFT'
) -> Tuple[np.array]:
    """
    Executa o pipeline completo de registro de imagens.

    Etapas:
        1. Conversão para escala de cinza.
        2. Detecção de keypoints.
        3. Extração de descritores.
        4. Correspondência dos descritores.
        5. Teste da razão de Lowe.
        6. Estimação da homografia com RANSAC.
        7. Transformação projetiva.
        8. Construção do panorama.
        9. Desenho das correspondências.

    Parâmetros de entrada:
        -img1 : Primeira imagem.
        -img2 : Segunda imagem.
        -limiar: limiar definido pelo usuário (default 0.75)
        -descriptor: Descritor utilizado (SIFT, SURF, ORB ou BRIEF)
    Saídas da função:
        -panoram: Imagem panorâmica resultante.
        -cors: Imagem com as correspondências encontradas.
    """
    descriptor = descriptor.upper()

    # (1) Converter as imagens coloridas de entrada em imagens de níveis de cinza.
    img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # (2) Encontrar pontos de interesse e descritores invariantes locais para o par de imagens.
    kps1, descs1 = detect_and_descript(img1_g, descriptor)
    kps2, descs2 = detect_and_descript(img2_g, descriptor)

    # (3) computar distâncias (similaridades) entre cada descritor das duas imagens.
    matches, corr = compute_descriptor_similarity(descs1, descs2)

    # (4) selecionar as melhores correspondências para cada descritor de imagem.
    best_pairs = get_best_pairs(matches, corr, limiar)

    # (5) executar a técnica RANSAC (RANdom SAmple Consensus) para estimar a matriz de homografia
    H, mask = estimate_homography(kps1, kps2, best_pairs)

    # (6) aplicar uma projeção de perspectiva (cv2.warpPerspective) para alinhar as imagens.
    perspective, dx, dy = get_perspective(img1, img2, H)

    # (7) unir as imagens alinhadas e criar a imagem panorâmica.
    panoram = get_panoram(img2, perspective, dx, dy)

    # (8) desenhar retas entre pontos correspondentes no par de imagens.
    cors = draw_correspondencies(img1, img2, kps1, kps2, best_pairs, mask)

    return panoram, cors


def get_processing_time(images_path = 'images'):
    """
    Calcula o tempo de processamento de cada técnica

    Parâmetros de entrada:
        -images_path : Caminho das imagens de entrada
    Saídas da função:
        -processing_times: Tempos de processamento
    """

    imgs_a = glob.glob(os.path.join(images_path, '*A.jpg'))
    imgs_b = glob.glob(os.path.join(images_path, '*B.jpg'))


    processing_times = {"SIFT": [], "ORB": [], "BRIEF": []}
    for descriptor in ["SIFT", "ORB", "BRIEF"]:
        for img_a_p, img_b_p in zip(imgs_a, imgs_b):
            print(descriptor)
            print(img_a_p, img_b_p)
            img_a = cv2.imread(img_a_p)
            img_b = cv2.imread(img_b_p)
            start = time.time()
            _, _ = registry(
                img_a,
                img_b,
                limiar=0.75,
                descriptor=descriptor
            )
            end = time.time()
            print(f'processing time: {end - start} s')
            print('-------------------------------')
            processing_times.setdefault(descriptor, []).append(end - start)

    return processing_times


def main():
    start = time.time()
    args = read_args()


    img1 = cv2.imread(args.i1)
    img2 = cv2.imread(args.i2)


    nome1 = os.path.basename(args.i1).split('.')[0]
    nome2 = os.path.basename(args.i2).split('.')[0]
    img_name = f"{nome1}_{nome2}"

    panoram, cors = registry(
        img1,
        img2,
        limiar=args.l,
        descriptor=args.d
    )

    os.makedirs(args.o, exist_ok=True)
    panorama_path = os.path.join(args.o, f'{img_name}_{args.l}_{args.d}_panoram.jpg')
    correspondencies_path = os.path.join(args.o, f'{img_name}_{args.l}_{args.d}_correspondencies.jpg')
    cv2.imwrite(panorama_path, panoram)
    cv2.imwrite(correspondencies_path, cors)


    end = time.time()
    print(f'processing time: {end - start} s')


if __name__ == "__main__":
    main()
