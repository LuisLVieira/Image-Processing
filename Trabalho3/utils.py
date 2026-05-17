import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


def plot_spectra(
        image: np.array,
        spectra: np.array,
        cent_spectra: np.array,
        title: str,
        output_path: str='./outputs'
) -> None:
    """
    Produz plots e salva os espectros de Fourier
        Parâmetros de entrada:
            - image: imagem monocromética ou colorida
            - spectra: Magnitude do espectro de Fourier em escala
                loagaritmica
            - cent_spectra: Magnitude do espectro de Fourier em escala
            logarítmica centrada (baixas frequências no centro)
    """

    os.makedirs(output_path, exist_ok=True)

    fig, axs = plt.subplots(1, 3, figsize=(10,5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title(f'{title} original')
    axs[0].axis(False)

    axs[1].imshow(spectra)
    axs[1].set_title(f'FFT {title}')
    axs[1].axis(False)

    axs[2].imshow(cent_spectra)
    axs[2].set_title(f'FFT centrada {title}')
    axs[2].axis(False)

    # Salva o espectro normalizado
    cv2.imwrite(
        f'{output_path}/fft_{title.split('.')[0]}.png',
        (255 * cent_spectra / cent_spectra.max()).astype(np.uint8)
    )

    # Mostra os componentes da transformada de Fourier
    plt.tight_layout()
    plt.savefig(
        f'{output_path}/{title.split('.')[0]}.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


def get_fft_spectra(
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
    title = os.path.basename(image_path)

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


    return img, mag_fft_log, mag_fft_cent_log


