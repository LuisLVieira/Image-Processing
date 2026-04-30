import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from typing import List
import os
import math


def calculate_grayscale_histogram(image: np.array, normalized: bool = True):
    """
    Calcula o histograma de uma imagem
    Parâmetros de entrada:
        - image: imagem monocromárica.
        - normalized: Se verdadeiro, normaliza o histograma (probabilidades)
    Saída da função:
        histograma da imagem
    """
    hist = np.bincount(image.flatten(), minlength=256)
    if normalized:
        hist = hist / hist.sum()

    return hist


def plot_limarization(
    original_image: np.array,
    segmented_image: np.array,
    title: str = "",
    figsize: tuple = (15, 5),
    save_dir: str = 'outputs/',
    threshold: str = ''
):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok = True)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    axs[0, 0].imshow(original_image, cmap="gray")
    axs[0, 0].set_title('Imagem original')
    axs[0, 0].axis(False)
    histogram = calculate_grayscale_histogram(original_image)
    axs[0, 1].bar(range(256), histogram, width=1.0)
    axs[0, 1].set_title('Histograma da imagem original')
    axs[0, 1].set_xlabel("Intensidade")
    axs[0, 1].set_ylabel("Frequência")
    axs[0, 1].set_xlim([0, 255])
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

    if threshold:
        threshold = int(threshold)
        axs[0, 1].axvline(
            x=threshold,
            linestyle="--",
            linewidth=2,
            color = 'r'
        )

        ymax = max(histogram)
        axs[0, 1].annotate(
            f"T = {threshold}",
            xy=(threshold, ymax * 0.9),
            xytext=(threshold + 5, ymax * 0.95),
            fontsize=15,
            color = 'r'
        )

    black_pixels = np.sum(segmented_image == 0)
    total_pixels = segmented_image.size
    percent_black = (black_pixels / total_pixels)*100
    percent_white = 100 - percent_black

    black_pixels_percent = f'{percent_black:.2f}%'
    white_pixels_percent = f'{percent_white:.2f}%'

    segmented_histogram = calculate_grayscale_histogram(
        segmented_image,
        normalized=True
    )

    axs[1, 1].annotate(
        black_pixels_percent,
        xy=(5, segmented_histogram[0]),
        xytext=(5 + 0.1, segmented_histogram[0]-0.05),
        fontsize=15,
    )

    axs[1, 1].annotate(
        white_pixels_percent,
        xy=(250, segmented_histogram[-1]),
        xytext=(250 - 50, segmented_histogram[-1]-0.05),
        fontsize=15,
    )

    # Mostra cada imagem transformada
    axs[1, 0].imshow(segmented_image, cmap="gray")
    axs[1, 0].set_title("Imagem segmentada")
    axs[1, 0].axis(False)
    axs[1, 1].bar(range(256), segmented_histogram, width=5)
    axs[1, 1].set_title('Histograma após segmentar')
    axs[1, 1].set_xlabel("Intensidade")
    axs[1, 1].set_ylabel("Frequência")
    axs[1, 1].set_xlim([0, 255])
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

    cv2.imwrite(
        f'{save_dir}/{title.lower().replace(' ', '_')}.png',
        segmented_image
    )

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f'{save_dir}/histograma_{title.lower().replace(' ', '_')}.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


def plot_images(
        images: List[np.array],
        title = "",
        save_dir: str = 'outputs/',
        fig_size=(8,8)
):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok = True)
    fig, axs = plt.subplots(3, 3, figsize=fig_size)
    for i, image in enumerate(images):
        axs = axs.flatten()

        axs[i].imshow(image, cmap="gray")
        axs[i].axis(False)

    for i in range(len(images), len(axs)):
        axs[i].axis('off')

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(
        f'{save_dir}/{title.lower().replace(' ', '_')}.png',
        bbox_inches='tight',
        dpi=300
    )
    plt.show()


class GlobaLimiarization:
    def __init__(self, image: np.ndarray):
        """
        Classe para limiarização global. Pixels maiores que um limiar
        são classificados como objeto (preto) e menores como fundo (branco)
        Parâmetros de entrada:
            - image: Imagem monocromática.
        """
        self.image = image

    def global_limiarization(self, T: int):
        """
        Aplica a limiarização binária de global uma imagem monocromática por um
        limiar.
        Parâmetros de entrada:
            - threshold: limiar para a limiarização binária.
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        # Trunca o valor do limiar para o intervalo [0, 255]
        threshold = np.clip(T, 0, 255)

        # Aplica a limiarização global
        # pixels com valor acima do limiar recebem o valor preto (0)
        # pixels com valor abaixo ou igual ao limiar recebem o valor branco (255)
        return np.where(self.image > threshold, 0, 255).astype(np.uint8)

    def otsu_limiarization(self) -> np.array:
        """
        Aplica a limiarização binária de global ao descobrir um limiar em que
        a variância entre objeto e fundo é máxima.
        Parâmetros de entrada:
            - threshold: limiar para a limiarização binária.
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        # Calcula o histograma em probabilidades
        histogram = calculate_grayscale_histogram(self.image, normalized=True)

        # Probabilidade acumulada (soma acumulada do histograma) para cada T (0 a 255)
        omega = np.cumsum(histogram)
        # Média acumulada
        mu = np.cumsum(histogram * np.arange(256))

        # média global
        mu_t = mu[-1]

        # Fórmula de máxima variância entre as classes
        # 1e-12 evita divisão por 0
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)

        # Threshold ótimo
        threshold = np.argmax(sigma_b2)

        # Imagem binária
        binary = np.where(self.image > threshold, 0, 255).astype(np.uint8)

        return binary, threshold

    def otsu_limiarization_cv(self) -> np.array:
        """
        Aplica a limiarização binária de global ao descobrir um limiar em que
        a variância entre objeto e fundo é máxima. Usa o OpenCV
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        threshold, binary = cv2.threshold(
            self.image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU # calcula o limiar de otsu e atribui 0 para > limiar
        )

        return binary, threshold


class LocalLimiarization:
    def __init__(self, image: np.ndarray, window_size: int):
        """
        Classe para limiarização local. Pixels maiores que um limiar baseado na
        vizinhança são classificados como objeto (preto) e menores como fundo (branco)
        Parâmetros de entrada:
            - image: Imagem monocromática.
            - window_size: tamanho da janela da vizinhança
        """
        self.image = image
        self.window_size = window_size

        # Padding das bordas por replicação
        padded = np.pad(
            self.image,
            self.window_size // 2,
            mode="edge"
        )

        # Obtenção das janelas da imagem
        # Formato (H, W, window_size, window_size)
        self.windows = sliding_window_view(
            padded,
            (self.window_size, self.window_size)
        )

        # Cálculo da média e desvio padrão da vizinhança (dentro da janela)
        self.mean = self.windows.mean(axis=(-2, -1))
        self.std = self.windows.std(axis=(-2, -1))


    def bernsen_limiarization(self):
        """
        Aplica a limiarização binária local de Bersen, com o limiar sendo
        o máximo menos o mínimo da região
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        local_min = self.windows.min(axis=(-2, -1))
        local_max = self.windows.max(axis=(-2, -1))

        threshold = (local_max + local_min) / 2

        return np.where(self.image >= threshold, 0, 255).astype(np.uint8)

    def niblack_limiarization(
        self,
        k: float = 0.2
    ):
        """
        Aplica a limiarização binária local de Niblack
        Parâmetros de entrada:
            - k: constante para desvio padrão.
        Saída da função:
            Imagem transformada com a limiarização binária.
        """

        threshold = self.mean + k * self.std

        return np.where(self.image >= threshold, 0, 255).astype(np.uint8)

    def sauvola_limiarization(
        self,
        k: float = 0.5,
        R: int = 128
    ):
        """
        Aplica a limiarização binária local de Sauvola e Pietaksinen
        Parâmetros de entrada:
            - k: constante para desvio padrão.
            - R: constante que divide o desvio padrão
        Saída da função:
            Imagem transformada com a limiarização binária.
        """

        threshold = self.mean * (1 + k * ((self.std / R) - 1))

        return np.where(self.image >= threshold, 0, 255).astype(np.uint8)

    def phansalskar_limiarization(
        self,
        k: float = 0.25,
        R: float = 0.5,
        p: int = 2,
        q: int = 10
    ):
        """
        Aplica a limiarização binária local de Phansalskar
        Parâmetros de entrada:
            - k: constante para desvio padrão.
            - R: constante que divide o desvio padrão
            - p: constante da fórmula
            - q: constante da fórmula
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        # Necessária a normalização da imagem
        mean = self.mean / 255.0
        std = self.std / 255.0
        threshold = mean * (
            1
            + p * np.exp(-q * mean)
            + k * ((std / R) - 1)
        )

        normalized_image = self.image.astype(np.float32) / 255.0

        return np.where(normalized_image >= threshold, 0, 255).astype(np.uint8)

    def contrast_limiarization(self):
        """
        Aplica a limiarização binária local por contraste, sendo preto quando o limiar
        está mais próximo do máximo e branco caso contrário.
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        # Calcula as distâncias para o máximo e mínimo local (presentes nas últimas dimenões = Janela)
        local_min = self.windows.min(axis=(-2, -1))
        local_max = self.windows.max(axis=(-2, -1))
        dist_to_min = np.abs(self.image - local_min)
        dist_to_max = np.abs(local_max - self.image)

        return np.where(dist_to_max < dist_to_min, 0, 255).astype(np.uint8)

    def mean_limiarization(self, C: float):
        """
        Aplica a limiarização binária local por contraste, com o limiar sendo
        a média dos vizinhos menos uma constante.
        Parâmetros de entrada:
            - C: constante.
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        threshold = np.clip(self.mean - C, 0, 255)

        return np.where(self.image >= threshold, 0, 255).astype(np.uint8)

    def median_limiarization(self):
        """
        Aplica a limiarização binária local por contraste, com o limiar sendo
        a mediana da vizinhança.
        Saída da função:
            Imagem transformada com a limiarização binária.
        """
        self.median = np.median(self.windows, axis=(-2, -1))

        return np.where(self.image >= self.median, 0, 255).astype(np.uint8)


class Videos:
    def __init__(self, video_path: str):
        """
        Métodos para processamento de quadros de vídeos.
        Parâmetros de entrada:
            - video_path: caminho do vídeo de entrada.
        """
        self.video_path = video_path
        self.video_name = os.path.basename(self.video_path).split('.')[0]
        self.video_cap = cv2.VideoCapture(video_path)
        # Reseta o video toda vez que inicia a classe
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def show_video(self):
        """
        Mostra na tela o vídeo.
        """
        # Reseta o video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Percorre todos os frames do vídeo (conjunto de imagens)
        while True:
            ret, frame = self.video_cap.read()
            # Se não retorna nada, chegou ao final do vídeo
            if not ret:
                break
            cv2.imshow("Video", frame)
            # Mostra o vídeo em 40 fps (25ms)
            # Interrompe o vídeo com a tecla q
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Fecha o vídeo
        self.video_cap.release()
        cv2.destroyAllWindows()


    def save_video(
        self,
        transition_indexes,
        title: str = '',
        output_path: str = 'outputs/videos',
        fps: int=30
    ):
        """
        Salva vídeos processados.
        Parâmetros de entrada:
            - frames: Quadros do vídeo a ser salvo.
            - title: nome do arquivo
            - output_path: caminho para salvar
            - fps: quadros por segundo
        """
        os.makedirs(output_path, exist_ok=True)
        # Reseta o video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Le os frames do vídeo
        ret, frame = self.video_cap.read()
        # Cálculo das dimensões das imagens que compõem o vídeo
        h, w = frame.shape[0:2]
        # Nome do arquivo salvo
        filename = f"{self.video_name}_{title}.mp4"
        # Reseta o video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Configurações para salvar em .mp4 mantendo as dimenções com fps definido
        result = cv2.VideoWriter(
            os.path.join(output_path, filename),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (w,h)
        )
        summary = []
        counter = 0
        while True:
            ret, frame = self.video_cap.read()
            # Se não retorna nada, chegou ao final do vídeo
            if not ret:
                break
            transition_indexes = set(transition_indexes)
            if counter in transition_indexes:
                result.write(frame)
                summary.append(frame)
            counter +=1

        # Fecha o vídeo
        result.release()
        cv2.destroyAllWindows()

        return summary

    def pixel_diff(self, T1, T2):
        """
        Detecta transições abruptas entre vídeos pela diferença entre pixels
        Parâmetros de entrada:
            - T1: Limiar de diferença entre pixels.
            - T2: Limiar de número de pixels diferentes
        Saídas da função:
            - transition_frames: frames de regiões de transição (vídeo sumarizado).
            - indexes: índices de transição
            - measures: valores da métrica medida (soma dos pixels diferentes)
        """
        transition_frames = []
        indexes = []
        measures = []

        # Reseta o video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Recupera o primeiro quadro do vídeo
        ret, frame = self.video_cap.read()
        # Converte para escala de cinza
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)
        counter = 0
        while True:
            # recupera o segunfo quadro
            ret, frame2 = self.video_cap.read()
            # encerra ao final do vídeo (sem resposta)
            if not ret:
                break
            # Converte o segundo quadro para escala de cinza
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(int)

            # Diferença entre quadros anterior e atual
            diff = np.abs(frame - frame2)

            # Total de pixels com difereça maior que o limiar
            total_diff = np.sum(diff >= T1)

            # Salva todas as medidas
            measures.append(total_diff)

            # Detecção de transição abrupta.
            if total_diff >= T2:
                indexes.append(counter)

            # Atualiza o próximo fram
            counter +=1
            frame = frame2

        return indexes, measures


    def block_diff(self, T1, T2, n_blocks=8):
        """
        Detecta transições abruptas entre vídeos pela diferença entre blocos de pixels
        Parâmetros de entrada:
            - T1: Limiar de diferença entre blocos.
            - T2: Limiar de número de blocos diferentes
            - n_blocks: número de blocos (Padrão 8x8, mas pode 16x16, etc)
        Saídas da função:
            - transition_frames: frames de regiões de transição (vídeo sumarizado).
            - indexes: índices de transição
            - measures: valores da métrica medida (soma dos pixels diferentes)
        """
        indexes = []
        measures = []

        # Reseta o video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Recupera o primeiro quadro do vídeo
        ret, frame = self.video_cap.read()
        # Converte para escala de cinza
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)
        # Calcula a altura e largura de cada bloco (divisões)
        block_height = frame.shape[0] // n_blocks
        block_width = frame.shape[1] // n_blocks

        counter = 0
        while True:
            # recupera o segundo quadro
            ret, frame2 = self.video_cap.read()
            # encerra ao final do vídeo (sem resposta)
            if not ret:
                break
            # Converte o segundo quadro para escala de cinza
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(int)

            # Separação da imagem em blocos de forma vetorizada
            # (n_blocks_x, block_height, n_blocks_y, block_width)
            blocks = frame.reshape(n_blocks, block_height, n_blocks, block_width)

            # Trasposição para o formato:
            # (n_blocks, n_blocks, block_height, block_width)
            blocks = blocks.transpose(0, 2, 1, 3)

            # Separação da imagem em blocos de forma vetorizada
            # (n_blocks_x, block_height, n_blocks_y, block_width)
            blocks2 = frame2.reshape(n_blocks, block_height, n_blocks, block_width)

            # Trasposição para o formato:
            # (n_blocks, n_blocks, block_height, block_width)
            blocks2 = blocks2.transpose(0, 2, 1, 3)

            # Erro quadrático médio por bloco (nas últimas dimensões estão as dimensões dos blocos)
            mse_blocks = np.mean(
                (blocks2.astype(np.float32) - blocks.astype(np.float32))**2,
                axis=(-1,-2)
            )

            # Detecção de blocos de transição por T1
            transition_blocks = np.sum(mse_blocks >= T1)

            # Salva todas as medidas de transição
            measures.append(transition_blocks)

            # Detecção de transição abrupta por T2.
            if transition_blocks > T2:
                indexes.append(counter)

            # Atualiza o próximo frame
            counter +=1
            frame = frame2

        return indexes, measures


    def hist_diff(self, alpha=4):
        """
        Detecta transições abruptas entre vídeos pela diferença entre os histogramas
        dos frames
        Parâmetros de entrada:
            - alpha: Limiar de diferença entre blocos
        Saídas da função:
            - transition_frames: frames de regiões de transição (vídeo sumarizado).
            - indexes: índices de transição
            - measures: valores da métrica medida (soma dos pixels diferentes)
        """
        indexes = []
        measures = []
        # Reseta o video
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        # Recupera o primeiro quadro do vídeo
        ret, frame = self.video_cap.read()
        # Converte para escala de cinza
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)
        counter = 0
        while True:
            # recupera o segundo quadro
            ret, frame2 = self.video_cap.read()
            # encerra ao final do vídeo (sem resposta)
            if not ret:
                break
            # Converte o segundo quadro para escala de cinza
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(int)

            # Calcula o histograma de cada quadro em probabilidade
            hist1 = calculate_grayscale_histogram(frame, normalized=True)
            hist2 = calculate_grayscale_histogram(frame2, normalized=True)

            # Cálculo da diferença total absoluta entre os histogramas
            diff = np.sum(np.abs(hist1 - hist2))

            # Salva todas as medidas de transição
            measures.append(diff)


            # Atualiza o próximo fram
            counter +=1
            frame = frame2

        # Calcula a média e o desvio padrão das diferenças
        mean_diff = np.mean(measures)
        std_diff = np.std(measures)

        # Limiar é definido pela média e um alpha ponderando o desvio padrão
        T = mean_diff + alpha*std_diff

        # Transições quando a diferença absoluta é maior que T (+ 1 pois considera frame 2 o momento da mudança)
        detected = np.where(np.array(measures) >= T)[0] + 1
        # Obtenção dos frames sumarizados
        indexes = detected.tolist()

        return indexes, measures, T


    def plot_measures(self, measures, T, title, output_path='outputs/videos'):
        """
        Mostra os valores das medidas pelo tempo
        """
        fig = plt.figure(figsize=(8,8))
        plt.title(f"{title} {self.video_name}")
        plt.xlabel("Quadros")
        plt.ylabel("Medida")
        plt.plot(list(range(0, len(measures), 1)), measures, '-k')
        plt.grid(linestyle='--')
        plt.axhline(y= T, color='r', linestyle='-', alpha=0.8)
        plt.savefig(f'{output_path}/{title}_{self.video_name}.png', dpi=300, bbox_inches="tight")
        plt.show()


    def plot_summary_images(
        self,
        summary,
        title,
        output_path="outputs",
        cols=4
    ):
        """
        Mostra alguns quadros resultantes da sumarização
        Parâmetros de entrada:
            - summary: lista de quadros
            - title: nome do gráfico
            - output_path: caminho para salvar
            - cols: número de colunas da grade
        """

        if len(summary) == 0:
            print("Nenhuma imagem para mostrar.")
            return

        os.makedirs(output_path, exist_ok=True)

        rows = math.ceil(len(summary) / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(16, 4 * rows))

        if rows == 1:
            axs = np.array(axs).reshape(1, -1)

        axs = axs.flatten()

        for i, img in enumerate(summary):
            if len(img.shape) == 3:
                img_plot = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axs[i].imshow(img_plot)
            else:
                axs[i].imshow(img, cmap="gray")

            axs[i].axis("off")

        for j in range(len(summary), len(axs)):
            axs[j].axis("off")

        filename = title.replace(' ', '_') + '.png'
        plt.suptitle(title + f' {self.video_name}', fontsize=16, y=1.02)

        plt.tight_layout()

        plt.savefig(
            os.path.join(output_path, self.video_name + filename),
            dpi=300,
            bbox_inches="tight"
        )

        plt.show()