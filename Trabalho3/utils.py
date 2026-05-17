import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.signal import find_peaks


def get_fft_spectrum(
        image_path: str
) -> np.array:
    """
        Obtem o espectro de Fourier centralizado de uma imagem.
        Parâmetros de entrada:
            - image_path: caminho de uma imagem monocromética ou colorida
        Saídas da função:
            - img: imagem original
            - mag_fft_log: Magnitude do espectro de Fourier em escala
                loagaritmica
            - mag_fft_cent_log: Magnitude do espectro de Fourier em escala
            logarítmica centrada (baixas frequências no centro)
    """

    img = cv2.imread(image_path)

    # convertendo imagens coloridas para escala de cinza
    # Suficiente para identificar estruturas na imagem
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Obtendo a dft complexa
    fft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)

    # magnitude da fft
    mag_fft = np.sqrt(fft[:,:,0]**2 + fft[:,:,1]**2)

    # escala logarítmica (+1 evita log(0))
    mag_fft_log = np.log(mag_fft + 1)

    # Obtendo a fft centrada
    cent_fft = np.fft.fftshift(fft)

    # magnitude da fft centrada
    mag_fft_cent = np.sqrt(cent_fft[:,:,0]**2 + cent_fft[:,:,1]**2)

    # escala logarítmica (+1 evita log(0))
    mag_fft_cent_log = np.log(mag_fft_cent + 1)

    # normalização do espectro
    mag_fft_cent_log = (255*mag_fft_cent_log/mag_fft_cent_log.max())

    return img, mag_fft_log, mag_fft_cent_log


def plot_spectrum(
        image: np.array,
        spectrum: np.array,
        cent_spectrum: np.array,
        title: str,
        output_path: str='./outputs'
) -> None:
    """
    Produz plots e salva os espectros de Fourier
        Parâmetros de entrada:
            - image: imagem monocromética ou colorida
            - spectrum: Magnitude do espectro de Fourier em escala
                loagaritmica
            - cent_spectrum: Magnitude do espectro de Fourier em escala
            logarítmica centrada (baixas frequências no centro)
    """

    os.makedirs(output_path, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f'{title} original')
    axs[0].axis(False)

    axs[1].imshow(spectrum)
    axs[1].set_title(f'FFT {title}')
    axs[1].axis(False)

    axs[2].imshow(cent_spectrum)
    axs[2].set_title(f'FFT centralizada {title}')
    axs[2].axis(False)

    # Salva o espectro normalizado
    cv2.imwrite(
        f'{output_path}/fft_{title.split('.')[0]}.png',
        (255 * cent_spectrum / cent_spectrum.max()).astype(np.uint8)
    )

    # Mostra os componentes da transformada de Fourier
    plt.tight_layout()
    plt.savefig(
        f'{output_path}/{title.split('.')[0]}.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


def get_spectrum_energy_histogram(
        spectrum: np.array,
        bins: int,
        exclude_ratio: int
) -> Tuple[np.array, np.array]:
    """
        Calcula o histograma de energia por ângulo com relação à origem,
        do espectro bidimensional gerado na transformada de Fourier
        (distribuição angular de energia).
        Parâmetros de entrada:
            - spectrum: espectro bidimencional centralizado
            - bins: número de bins do histograma
        Saídas da função:
            - hist: Intensidades do histograma para cada bin
            - bin_centers: ângulos em radianos de cada bin
            - bin_centers_deg: ângulos em graus de cada bin
    """

    # Obtenção do ponto central da imagem (referência)
    xc, yc = spectrum.shape[1]//2, spectrum.shape[0]//2

    # Obtenção do indices de cada pixel do espectro
    rows, columns = np.indices(spectrum.shape)

    # Obtenção dos indices relativos ao ponto central (centrado no (0, 0))
    x = columns - xc
    y = rows - yc

    # Obtenção dos angulos de cada pixel com relação ao centro pela fórmula
    # angulo = arctg((y-0)/(x-0))
    # usa arctan2 para obter angulos de [-pi, pi], distinguindo o quadrante
    angles = np.arctan2(y, x)

    # Exclusão dos valores muito próximos ao centro (baixas frequências)
    # Distância euclidiana de cada ponto ao centro
    # Só considera os pixels de índices maiores que o raio definido
    dists = np.sqrt(x**2 + y**2)
    angles = angles[dists > exclude_ratio]
    spectrum = spectrum[dists > exclude_ratio]

    # Cálculo do histograma
    hist, bin_edges = np.histogram(
        angles,
        bins=bins,
        weights = spectrum
    )

    # Array central de cada bin
    # np.histogram retorna as bordas dos bins
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def get_dominant_angles(hist, bin_centers) -> list:
    """
        Retorna os ângulos dominantes a partir do histograma
        de energia, que possuem máxima energia.
        Parâmetros de entrada:
            - hist: histograma
            - bin_centers: valores dos bins
        Saídas da função:
            - angles: ângulos dominantes

    """
    indices,_ = find_peaks(
        hist,
        prominence=0.1*hist.max()
    )

    return bin_centers[indices]


def plot_spectrum_energy_histogram(
    image: np.array,
    spectrum: np.array,
    bin_centers: np.array,
    hist: np.array,
    title: str,
    output_path: str = './outputs',
) -> None:
    """
        Mostra a imagem original, o histograma obtido e o espectro
        de Fourier, destacando as orientações dominantes.
        Parâmetros de entrada:
            - image: imagem original
            - spectrum: espectro de fourier da imagem
            - bin_centers: ângulos centrais de cada bin
            - hist: histograma de energia por ângulo
    """
    os.makedirs(output_path, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f'{title} original')
    axs[0].axis(False)

    axs[1].imshow(spectrum)
    axs[1].set_title(f'FFT {title}')
    axs[1].axis(False)

    dominant_angles = get_dominant_angles(hist, bin_centers)

    # Mostra na imagem a reta associada a cada ângulo
    # O raio R pode ser definido arbitrariamente
    # Foi utilizada a metade distância do centro
    R = min(spectrum.shape)/2
    # centro do espectro
    xc, yc = spectrum.shape[1]//2, spectrum.shape[0]//2
    print(dominant_angles)
    for angle in dominant_angles:
        # dado o ângulo, utiliza-se as equações para as coordenadas
        # x = R*cos(angle)
        # y = R*sen(angle)
        x = R*np.cos(angle)
        y = -R*np.sin(angle) # negativo para adaptar ao imshow

        # coordenandas de 2 pontos da reta a ser desenhada
        x1, y1 = xc + x, yc + y
        x2, y2 = xc - x, yc - y

        # plot da reta
        axs[1].plot(
            [x1, x2],
            [y1, y2],
            color = 'cyan',
            linestyle='-',
            linewidth=2
        )

        axs[1].annotate(
            f'{np.degrees(angle):.2f}°',
            (x1, y1),
            color='cyan',
            fontsize=12
        )

    axs[2].plot(bin_centers, hist)
    axs[2].set_title(f'histograma angular {title}')
    axs[2].set_xlabel('bin angular (rad)')
    axs[2].set_ylabel('energia')

    # Mostra os componentes da transformada de Fourier
    plt.tight_layout()
    plt.savefig(
        f'{output_path}/histogram_{title.split('.')[0]}.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.show()
