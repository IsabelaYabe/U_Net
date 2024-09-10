import os
import random, cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PreprocessDataset(Dataset):
    def __init__(self, dir_dataset, df_labels, dir_new_dataset, set_type="train", with_brightness_contrast=True, var_gaussian=10, amount = 0.02, prob_gen=0.4, partition=3): 
        self.with_brightness_contrast = with_brightness_contrast
        self.var_gaussian = var_gaussian
        self.amount = amount
        self.dir_dataset = dir_dataset
        self.prob_gen = prob_gen
        self.set_type = set_type
        self.dir_new_dataset = dir_new_dataset
        self.partition = partition
        self.rgb_to_index, self.index_to_label = self.dict_labels(df_labels)
        self.df_labels = df_labels
        self.dim_images = None
        self.num_images = None
        self.image_list = [] 
        self.label_list = [] 
        self.num_classes = len(self.df_labels)
        self.load_images_from_folder()
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # Carregar a imagem e o rótulo baseado no índice `idx`
        img_path = self.image_list[idx]
        lbl_path = self.label_list[idx]

        # Carregar a imagem e o rótulo do disco
        img = cv2.imread(img_path)
        lbl = cv2.imread(lbl_path)

        # Normalizar a imagem (opcional)
        try:
            img = img / 255.0
        except:
            print(f"Erro ao normalizar a imagem: {img_path}")
        
        lbl = self.rgb_to_label(lbl)


        # Converter os dados para tensores
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # De (H, W, C) para (C, H, W)
        lbl = torch.tensor(lbl, dtype=torch.long)  # (CH, W)

        return img, lbl

    def create_or_reset_directory(self):
        new_dir_img = os.path.join(self.dir_new_dataset, self.set_type)
        new_dir_lbl = os.path.join(self.dir_new_dataset, f"{self.set_type}_labels")
        
        # Verificar se o diretório de imagens já existe; se não, criar
        if not os.path.exists(new_dir_img):
            os.makedirs(new_dir_img)  # Cria o diretório se não existir
        
        # Verificar se o diretório de rótulos já existe; se não, criar
        if not os.path.exists(new_dir_lbl):
            os.makedirs(new_dir_lbl)  # Cria o diretório se não existir
        
    def load_images_from_folder(self):
        # Definir os caminhos originais dos datasets
        images_dir = os.path.join(self.dir_dataset, self.set_type)
        labels_dir = os.path.join(self.dir_dataset, f"{self.set_type}_labels")
        
        dim_images = self.load_images_and_labels(images_dir, labels_dir)
        self.dim_images = dim_images
            
    def load_images_and_labels(self, images_dir, labels_dir):
        self.create_or_reset_directory()

        # Listar arquivos nos diretórios
        image_files = sorted(os.listdir(images_dir))
        label_files = sorted(os.listdir(labels_dir))

        prob_to_transform = self.prob_gen
        img_lbl_files = zip(image_files, label_files)

        for img_name, lbl_name in img_lbl_files:
            transform_img = random.choices([True, False], weights=[prob_to_transform, 1 - prob_to_transform], k=1)[0]

            # Definir caminho para a imagem e rótulo
            img_path = os.path.join(images_dir, img_name)
            lbl_path = os.path.join(labels_dir, lbl_name)

            # Carregar imagem e rótulo
            img = cv2.imread(img_path)
            lbl = cv2.imread(lbl_path)

            # Reduzir a resolução da imagem
            img = img[::self.partition, ::self.partition, :]
            lbl = lbl[::self.partition, ::self.partition, :]
            
            # Converter de BGR para RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2RGB)

            # Aplicar Gaussian blur
            img = cv2.GaussianBlur(img, (3, 3), 0)

            # Transformação aleatória
            if transform_img:
                img_transform = img.copy()
                lbl_transform = lbl.copy()
                img_transform, lbl_transform, trans = self.apply_random_transformation(img_transform, lbl_transform)

                # Salvar imagem e rótulo transformado com sufixo de transformação
                transformed_img_name = f"{img_name}_{trans}.png"
                transformed_lbl_name = f"{lbl_name}_{trans}.png"

                processed_img_path = os.path.join(self.dir_new_dataset, self.set_type, transformed_img_name)
                processed_lbl_path = os.path.join(self.dir_new_dataset, f"{self.set_type}_labels", transformed_lbl_name)

                # Verifique se o arquivo já existe antes de salvar
                if not os.path.exists(processed_img_path):
                    cv2.imwrite(processed_img_path, img_transform)
                    cv2.imwrite(processed_lbl_path, lbl_transform)

                    # Adicionar as imagens e rótulos transformados às listas
                    self.image_list.append(processed_img_path)
                    self.label_list.append(processed_lbl_path)

            # Criar caminho para salvar a imagem transformada
            processed_img_path = os.path.join(self.dir_new_dataset, self.set_type, img_name)
            processed_lbl_path = os.path.join(self.dir_new_dataset, f"{self.set_type}_labels", lbl_name)

            cv2.imwrite(processed_img_path, img)
            cv2.imwrite(processed_lbl_path, lbl)
            
            # Adicionar as imagens e rótulos processados às listas
            self.image_list.append(processed_img_path)
            self.label_list.append(processed_lbl_path)

        return img.shape

    def apply_random_transformation(self, img, lbl):
        if self.with_brightness_contrast:
            random_transform = random.randint(0, 9)
        else:
            random_transform = random.randint(0, 4)
        if random_transform == 0:
            arg_1, arg_2 = self.augment(img, lbl, "rotation")
            return arg_1, arg_2, "rotation"
        elif random_transform == 1:
            arg_1, arg_2 = self.augment(img, lbl, "flip")
            return arg_1, arg_2, "flip"
        elif random_transform == 2:
            arg_1, arg_2 = self.augment(img, lbl, "flip_rotation")
            return arg_1, arg_2, "flip_rotation"
        elif random_transform == 3:
            arg_1, arg_2 = self.equalize_histogram(img, lbl)
            return arg_1, arg_2, "equalize_histogram"
        elif random_transform == 4:
            arg_1, arg_2 = self.add_noise(img, lbl, noise_type="gaussian")
            return arg_1, arg_2, "gaussian_noise"
        elif random_transform == 5:
            arg_1, arg_2 = self.add_noise(img, lbl, noise_type="salt_pepper")
            return arg_1, arg_2, "sal_pepper_noise"
        elif random_transform == 6:
            arg_1, arg_2 = self.adjust_brightness_contrast(img, lbl, brightness=0, contrast=50)
            return arg_1, arg_2, "b_c_0_50"
        elif random_transform == 7:
            arg_1, arg_2 = self.adjust_brightness_contrast(img, lbl, brightness=-10, contrast=30)
            return arg_1, arg_2, "b_c_-10_30"
        elif random_transform == 8:
            arg_1, arg_2 = self.adjust_brightness_contrast(img, lbl, brightness=60, contrast=60)
            return arg_1, arg_2, "b_c_60_60"
        else:
            arg_1, arg_2 = self.adjust_brightness_contrast(img, lbl, brightness=-20, contrast=0)
            return arg_1, arg_2, "b_c_-20_0"

    def rgb_to_label(self, mask):
        unique_colors = np.unique(mask.reshape(-1, mask.shape[2]), axis=0)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int32)
        for color in unique_colors:
            color_tuple = tuple(color)
            if color_tuple in self.rgb_to_index:
                label = self.rgb_to_index[color_tuple]
                matches = np.all(mask == color, axis=-1)
                label_mask[matches] = label
        return label_mask

    def dict_labels(self, df_labels):
        rgb_to_index = {}
        index_to_label = {}
        for count, row in df_labels.iterrows():
            color = (row['r'], row['g'], row['b'])
            label = row['name']
            rgb_to_index[color] = count
            index_to_label[count] = label
        return rgb_to_index, index_to_label

    def augment(self, img, lbl, transform):
        if transform == "rotation":
            rotation = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), 180, 1)
            rotated_image = cv2.warpAffine(img, rotation, (img.shape[1], img.shape[0]))
            rotated_label = cv2.warpAffine(lbl, rotation, (lbl.shape[1], lbl.shape[0]), flags=cv2.INTER_NEAREST)
            return rotated_image, rotated_label
        elif transform == "flip":
            flipped_image = cv2.flip(img, 1)
            flipped_label = cv2.flip(lbl, 1)
            return flipped_image, flipped_label
        elif transform == "flip_rotation":
            flipped_image = cv2.flip(img, 0)
            flipped_label = cv2.flip(lbl, 0)
            return flipped_image, flipped_label
        
   
    def equalize_histogram(self, img, lbl):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img, lbl
        
    def adjust_brightness_contrast(self, img, lbl, brightness=0, contrast=0):
        img = np.int16(img)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        return img, lbl
    
    def add_noise(self, img, lbl, noise_type="gaussian"):
        if noise_type == "gaussian":
            mean = 0
            var = self.var_gaussian
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, img.shape)
            noisy = img + gauss
            noisy = np.clip(noisy, 0, 255)
            return noisy, lbl
    
        elif noise_type == "salt_pepper":
            s_vs_p = 0.5
            amount = self.amount
            out = np.copy(img)
    
            # Salt mode
            num_salt = np.ceil(amount * img.size * s_vs_p)
            coords = [np.random.randint(0, i, int(num_salt)) for i in img.shape]
            out[tuple(coords)] = 255
    
            # Pepper mode
            num_pepper = np.ceil(amount * img.size * (1.0 - s_vs_p))
            coords = [np.random.randint(0, i, int(num_pepper)) for i in img.shape]
            out[tuple(coords)] = 0
    
            return out, lbl