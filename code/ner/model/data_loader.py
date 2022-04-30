import random
import numpy as np
import os
import sys

import torch
from torch.autograd import Variable

import utils 

class DataLoader(object):

    def __init__(self, data_dir, params):
        """
        ### Función `__init__`
        Inicializa la instancia de la clase, almacenando su vocabulario, diccionario de etiquetas,
        palabras desconocidas y padding.

        #### Parámetros:
        * `self`: instancia a inicializar.
        * `data_dir`: {str} ruta desde donde se leerá cada archivo para cargar.
        * `params`: {dict} instancia de Params con los parámetros del modelo.
        """

        # Genera la ruta desde la que se leerán los parámetros.
        json_path = os.path.join(data_dir, 'dataset_params.json')
        print(json_path)

        # Controla la excepción si no existe el archivo o no es tipo JSON.
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)

        # Almacena en la instancia los parámetros desde la ruta json_path.
        self.dataset_params = utils.Params(json_path)        

        # Genera la ruta desde la que se leerá el vocabulario.
        vocab_path = os.path.join(data_dir, 'words.txt')
        
        # Almacena en el vocabulario (tipo diccionario) en la instancia iterando línea por línea.
        self.vocab = {}
        with open(vocab_path) as f:
            for i, l in enumerate(f.read().splitlines()):
                self.vocab[l] = i
        
        # Almacena los ínidces de las palabras desconocidas y palabras padding en la instancia.
        self.unk_ind = self.vocab[self.dataset_params.unk_word]
        self.pad_ind = self.vocab[self.dataset_params.pad_word]
                
        # Lee y almacena las etiquetas que provengan del archivo tags.
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

        # Actualiza el diccionario params con los parámetros leídos desde el archivo json_path.
        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, d):
        """
        ### Función `load_sentences_labels`
        Almacena cada una de las palabras con sus respectivas etiquetas en el diccionario d.

        #### Parámetros:
        * `self`: instancia a inicializar.
        * `sentences_file`: {str} ruta desde donde se leerá el archivo que almacena las frases.
        * `labels_file`: {str} ruta desde donde se leerá el archivo que almacena las etiquetas.
        * `d`: {Params} instancia de Params con los parámetros del modelo.
        """

        sentences = []
        labels = []

        # Si el vocabulario contiene la palabra almacenada en la frase, entonces se almacena en el diccionario.
        # En caso contrario, se almacena la palabra desconocida en el array sentences.
        with open(sentences_file) as f:
            for sentence in f.read().splitlines():
                s = [self.vocab[token] if token in self.vocab 
                     else self.unk_ind
                     for token in sentence.split(' ')]
                sentences.append(s)

        # Almaena cada una de las etiquetas de cada palabra de cada frase y se almacenan en el array labels.
        with open(labels_file) as f:
            for sentence in f.read().splitlines():
                l = [self.tag_map[label] for label in sentence.split(' ')]
                labels.append(l)        

        # Se comprueba que el número de etiquetas y de palabras coinciden.
        assert len(labels) == len(sentences)
        for i in range(len(labels)):
            assert len(labels[i]) == len(sentences[i])

        # Asigna los parámetros al diccionario d.
        d['data'] = sentences
        d['labels'] = labels
        d['size'] = len(sentences)

    def load_data(self, types, data_dir):
        """
        ### Función `load_data`
        Dada una jerarquía de carpetas donde train, test y val se encuentren en el mismo nivel,
        procesa las frases y etiquetas de cada uno de los archivos almacenados.

        #### Parámetros:
        * `self`: instancia a inicializar.
        * `types`: {array} array que contará con el nombre de los distintos conjuntos de datos que se quieran procesar.
        * `data_dir`: {Params} directorio desde el que se leerán los archivos de frases y etiquetas.

        #### Return `data`: {dict} diccionario donde estarán almacenadas las frases y etiquetas de cada uno de los conjuntos de datos.
        """
        data = {}
        
        # Recorre cada conjunto de datos (entrenamiento, validación y test) y almacena en el diccionario data
        # la palabra y etiqueta de cada una de las frases correspondientes a su conjunto de datos.
        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file, data[split])

        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        ### Función `data_iterator`
        Permite seleccionar un conjunto de datos y iterar sobre ellos.

        #### Parámetros:
        * `self`: instancia a inicializar.
        * `data`: {dict} hash que almacena las frases y etiquetas de cada uno de los conjuntos de datos.
        * `params`: {Params} instancia de Params con los parámetros del modelo.
        * `shuffle`: {Boolean} parámetro que definirá si el orden de los datos
                               se tratan aleatoriamente o no.
        """

        # Si shuffle es True, entonces se ordena aleatoriamente el orden de las frases. 
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # Se almacena el número de frases y de palabras en el diccionario params.
        for i in range((data['size']+1)//params.batch_size):
            # Para cada uno de los batches se almacenan las frases y las etiquetas, incluyendo la posibilidad de aleatoriedad dentro del batch gracias al Shuffle,
            batch_sentences = [data['data'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]
            batch_tags = [data['labels'][idx] for idx in order[i*params.batch_size:(i+1)*params.batch_size]]

            # Crea una lista que sea el máximo de todas las frases del batch.
            batch_max_len = max([len(s) for s in batch_sentences])

            # Se crea una matriz de unos con el tamaño del minibatch y la frase más larga, tanto para las frases como para las etiquetas.
            batch_data = self.pad_ind*np.ones((len(batch_sentences), batch_max_len))
            # Donde no haya datos dentro del batch se asigna el valor -1.
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

            # Se rellena el minibatch con las frases y las etiquetas.
            for j in range(len(batch_sentences)):
                cur_len = len(batch_sentences[j])
                batch_data[j][:cur_len] = batch_sentences[j]
                batch_labels[j][:cur_len] = batch_tags[j]

            # Se transforman las frases y las etiquetas a un formato Tensor.
            batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

            # Si se dispone de GPU, se cargan los datos en la GPU.
            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()

            batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

            # Se devuelven los datos del minibatch en formato generator.
            yield batch_data, batch_labels
