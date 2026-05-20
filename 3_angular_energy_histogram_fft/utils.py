import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from typing import Tuple


def get_fft_spectrum(
        img: np.array
) -> np.array:
    """
        Obtem o espectro de Fourier centralizado de uma imagem.
        Parâmetros de entrada:
            - image: imagem monocromética ou colorida
        Saídas da função:
            - img: imagem original
            - mag_fft_log: Magnitude do espectro de Fourier
            em escala logarítmica.
            - mag_fft_cent_log: Magnitude do espectro de Fourier
            em escala logarítmica.
            centrada (baixas frequências no centro)
    """

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


    return img, mag_fft_log, mag_fft_cent_log


def get_fft_hann_spectrum(
        img: np.array
) -> np.array:
    """
        Obtem o espectro de Fourier centralizado de uma imagem.
        Com aplicação de janela de Hann para suavização
        Parâmetros de entrada:
            - image: imagem monocromética ou colorida
        Saídas da função:
            - img: imagem original
            - mag_fft_log: Magnitude do espectro de Fourier
            em escala logarítmica.
            - mag_fft_cent_log: Magnitude do espectro de Fourier
            em escala logarítmica.
            centrada (baixas frequências no centro)
    """

    # convertendo imagens coloridas para escala de cinza
    # Suficiente para identificar estruturas na imagem
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicação da janela de Hann para suavizar
    h, w = gray_img.shape

    hann_y = np.hanning(h)
    hann_x = np.hanning(w)

    hann2d = np.outer(
        hann_y,
        hann_x
    )

    windowed_img = gray_img * hann2d

    # Obtendo a dft complexa
    fft = cv2.dft(np.float32(windowed_img), flags=cv2.DFT_COMPLEX_OUTPUT)

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

    return img, mag_fft_log, mag_fft_cent_log


def plot_spectrum(
        image: np.array,
        spectrum: np.array,
        cent_spectrum: np.array,
        title: str,
        output_path: str='./outputs',
        show_raw_spectrum: bool = False
) -> None:
    """
    Produz plots e salva os espectros de Fourier
        Parâmetros de entrada:
            - image: imagem monocromética ou colorida
            - spectrum: Magnitude do espectro de Fourier em escala
                loagaritmica
            - cent_spectrum: Magnitude do espectro de Fourier
            centrada (baixas frequências no centro)
            - title: título do plot
            - output_path: diretório para salvar os arquivos
            - show_raw_spectrum: mostra o espectro nâo centralizado
    """

    os.makedirs(output_path, exist_ok=True)

    if show_raw_spectrum:
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10,5))

    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f'{title}')
    axs[0].axis(False)

    index = 1

    if show_raw_spectrum:
        axs[index].imshow(spectrum)
        axs[index].set_title(f'espectro {title}')
        axs[index].axis(False)

        index +=1

    axs[index].imshow(cent_spectrum)
    axs[index].set_title(f'espectro centralizado {title}')
    axs[index].axis(False)

    # Salva o espectro normalizado e em escala logarítmica
    # para melhor visualização
    cv2.imwrite(
        f'{output_path}/espectro_centralizado_{title}.png',
        (255 * cent_spectrum / cent_spectrum.max()).astype(np.uint8)
    )

    # Mostra os componentes da transformada de Fourier
    plt.tight_layout()
    plt.savefig(
        f'{output_path}/espectros_{title}.png',
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
            - exclude_ratio: raio de exclusão para baixa frequência
        Saídas da função:
            - hist: Intensidades do histograma para cada bin
            - bin_centers: ângulos em radianos de cada bin
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
        range=(-np.pi, np.pi),
        weights=spectrum
    )

    # Array central de cada bin
    # np.histogram retorna as bordas dos bins
    # Para encontrar o meio, basta obter os pontos médios
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return hist, bin_centers


def get_dominant_angles(
        hist: np.array,
        bin_centers: np.array,
        threshold_ratio: int=0.9
) -> list:
    """
        Retorna os ângulos dominantes a partir do histograma
        de energia, que possuem máxima energia. identificam
        ângulos que a diferença para o valor máximo estejam
        dentro da tolerância.
        Parâmetros de entrada:
            - hist: histograma
            - bin_centers: valores dos bins
            - threshold_ratio: Percentual de tolerância da
            diferança ao máximo para considerar um pico como
            dominante. Padrão é 0.9, ou seja, os picos dominantes
            são aqueles que tem intensidade maior que 0.9 do
            valor máximo.
        Saídas da função:
            - angles: ângulos dominantes

    """
    threshold = threshold_ratio * hist.max()

    indices = np.where(hist >= threshold)[0]

    print('Picos:')
    print(hist[indices])

    return bin_centers[indices]


def plot_spectrum_energy_histogram(
    image: np.array,
    spectrum: np.array,
    bin_centers: np.array,
    hist: np.array,
    title: str,
    tolerance: int = 0.9,
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
            - tolerance: Valor numérico de tolerância da
            diferança ao máximo para considerar um pico como
            dominante.
            - title: título do plot
            - output_path: diretório para salvar os arquivos
    """
    os.makedirs(output_path, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(title)
    axs[0].axis(False)

    axs[1].imshow(spectrum)
    axs[1].set_title(f'Espectro {title} (orientações dominantes)')
    axs[1].axis(False)

    dominant_angles = get_dominant_angles(hist, bin_centers, tolerance)

    # Mostra na imagem a reta associada a cada ângulo
    # O raio R pode ser definido arbitrariamente
    # Foi utilizada a metade distância do centro
    R = min(spectrum.shape)/2
    # centro do espectro
    xc, yc = spectrum.shape[1]//2, spectrum.shape[0]//2

    for angle in dominant_angles:
        # dado o ângulo, utiliza-se as equações para as coordenadas
        # x = R*cos(angle)
        # y = R*sen(angle)
        x = R*np.cos(angle)
        y = R*np.sin(angle)

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
            (x1 + 5, y1 + 5),
            color='cyan',
            fontsize=12
        )

    n_bins = len(list(bin_centers))

    axs[2].plot(list(range(n_bins)), hist)
    axs[2].set_title(f'histograma angular {title} {n_bins} bins')
    axs[2].set_xlabel('bin angular')
    axs[2].set_ylabel('energia')

    # Mostra os componentes da transformada de Fourier
    plt.tight_layout()
    plt.savefig(
        f'{output_path}/histograma_{title}_{n_bins}_bins.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


def translate_image(image: np.array, dx: int, dy: int) -> np.array:
    """
        Translada a imagem em x por um deslocamento dx e em y por dy.
        Utiliza BORDER_WRAP para um preenchimento circular.
        Parâmetros de entrada:
            - image: imagem original
            - dx: deslocamento no eixo x
            - dy: deslocamento no eixo y
        Saídas da função:
            - imagem transladada
    """

    h, w = image.shape[:2]

    # Matriz de translação
    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    # Aplica a translação
    return cv2.warpAffine(
        image,
        M,
        (w,h),
        borderMode=cv2.BORDER_WRAP
    )


def rotate_image(image: np.array, angle: int) -> np.array:
    """
        Rotaciona a imagem em ângulo especificado.
        Utiliza BORDER_CONSTANT para preencher com pixel preto.
        Parâmetros de entrada:
            - image: imagem original
            - rotation_angle: ângulo de rotação
        Saídas da função:
            - imagem rotacionada
    """
    h, w = image.shape[:2]

    # Centro da imagem
    center = (w/2,h/2)

    # Matriz de rotação
    M = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    # Aplica a rotação
    return cv2.warpAffine(
        image,
        M,
        (w,h),
        borderMode=cv2.BORDER_CONSTANT
    )


def scale_image(image: np.array, factor: int) -> np.array:
    """
        Escala a imagem por um fator especificado.
        Utiliza BORDER_CONSTANT para preencher com pixel preto.
        Parâmetros de entrada:
            - image: imagem original
            - factor: fator de escala
        Saídas da função:
            - imagem escalada
    """
    h, w = image.shape[:2]

    # Centro da imagem
    center = (w/2,h/2)

    # Matriz de escala
    M = cv2.getRotationMatrix2D(
        center,
        0,
        factor
    )

    # Aplica a escala
    return cv2.warpAffine(
        image,
        M,
        (w,h),
        borderMode=cv2.BORDER_CONSTANT
    )


def plots(
    img: np.array,
    title: str,
    output_path: str = './outputs',
    save_img: bool = True
) -> None:
    """
        Encapsula todos os plots desenvolvidos.
        obtidos.
        Parâmetros de entrada:
            - img: imagem de entrada
            - title: título do plot
            - output_path: diretório para salvar os arquivos
    """


    img, _, fft_mag_cent = get_fft_spectrum(img)

    title = title.replace('.png', '').replace('.pgm', '')

    # salva o espectro transladado
    if save_img:
        cv2.imwrite(
            f'{output_path}/{title}.png',
            img
        )

    title = title.replace('_', ' ')

    plot_spectrum(
        image=img,
        spectrum=fft_mag_cent,
        cent_spectrum=fft_mag_cent,
        title = title,
        output_path=output_path
    )

    hist, bin_centers = get_spectrum_energy_histogram(
        spectrum = fft_mag_cent,
        bins = 36,
        exclude_ratio = 60,
    )

    plot_spectrum_energy_histogram(
        image=img,
        spectrum=fft_mag_cent,
        bin_centers=bin_centers,
        hist=hist,
        title = title,
        tolerance=0.99,
        output_path=output_path
    )
