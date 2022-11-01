import numpy as np
import random
import carga

variable = []
numero = int(input("Ingrese el numero que quiere ver de predicciones"))


def sigmoidea(parametro):
    
    return 1.0/(1.0+np.exp(-parametro))


def primera_sigmoidea(parametro):
    
    return sigmoidea(parametro)*(1-sigmoidea(parametro))


class Red(object):

    def __init__(self, tamano):

        self.numero_capas = len(tamano)
        self.tamano = tamano
        self.sesgos = [np.random.randn(y, 1) for y in tamano[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(tamano[:-1], tamano[1:])]


    def retroalimentacion(self,entrada):

        for b, w in zip(self.sesgos, self.pesos):
            entrada = sigmoidea(np.dot(w, entrada)+b)

        return entrada
        

    def actualizacion_mini_lote(self, mini_lote, tasa_aprendisaje):

        nabla_b = [np.zeros(b.shape) for b in self.sesgos]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]

        for x, y in mini_lote:
            delta_nabla_b, delta_nabla_w = self.retropropagacion(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.pesos = [w-(tasa_aprendisaje/len(mini_lote))*nw for w, nw in zip(self.pesos, nabla_w)]
        self.sesgos = [b-(tasa_aprendisaje/len(mini_lote))*nb for b, nb in zip(self.sesgos, nabla_b)]


    def costo_derivado(self, salida_activaciones, y):

        return (salida_activaciones-y)


    def evaluar(self,datos_prueba):
        for (x, y) in datos_prueba:
            if len(x) != 784:
                print("paila")
            
        resultados_prueba = [(np.argmax(self.retroalimentacion(x)), y) for (x, y) in datos_prueba]
        
        i = 0
        while i < numero:
            variable.append(resultados_prueba[i])
            i += 1

        #Generar matrriz de aprobacion TASK
        diagonal = 0 
        matriz = [[0, 0, 0, 0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0]]
        
        num = [0,0,0,0,0,0,0,0,0,0]
        for (x, y) in resultados_prueba:
            num[y] += 1
            matriz[x][y] +=1
        

        i = 0
        j = 0 
        
        while i < 10:
            while j < 10:
                
                matriz[i][j] = round(matriz[i][j]/num[j],3)

                if i == j:
                    diagonal +=  matriz[i][j]
                j += 1
            i += 1
            j = 0

        i = 0
        #while i <10:
        #    print(matriz[i])
        #    i +=1
        print(diagonal)

        return sum(int(x == y) for (x, y) in resultados_prueba)

    def retropropagacion(self, x, y):
        
        nabla_b = [np.zeros(b.shape) for b in self.sesgos]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]
        
        activacion = x
        activaciones = [x] 
        zs = []

        for b, w in zip(self.sesgos, self.pesos):
            z = np.dot(w, activacion)+b
            zs.append(z)
            activacion = sigmoidea(z)
            activaciones.append(activacion)
        
        delta = self.costo_derivado(activaciones[-1], y) * \
            primera_sigmoidea(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activaciones[-2].transpose())
        
        for l in range(2, self.numero_capas):
            z = zs[-l]
            sp = primera_sigmoidea(z)
            delta = np.dot(self.pesos[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activaciones[-l-1].transpose())

        return (nabla_b, nabla_w)

    def SGD(self, datos_entrenamiento, epocas, mini_lote_tamano, tasa_aprendisaje, datos_prueba=None):
        """Train the neural network using mini-batch stochastic
        gradient descent, The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs, The other non-optional parameters are
        self-explanatory, If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out, This is useful for
        tracking progress, but slows things down substantially."""

        datos_entrenamiento = list(datos_entrenamiento)
        n = len(datos_entrenamiento)

        if datos_prueba:
            datos_prueba = list(datos_prueba)
            n_prueba = len(datos_prueba)

        for j in range(epocas):
            random.shuffle(datos_entrenamiento)
            mini_lotes = [ datos_entrenamiento[k:k+mini_lote_tamano] for k in range(0, n, mini_lote_tamano)]

            for mini_lote in mini_lotes:
                self.actualizacion_mini_lote(mini_lote, tasa_aprendisaje)

            if datos_prueba:
                print("Epoch {} : {} / {}".format(j,self.evaluar(datos_prueba),n_prueba))
            else:
                print("Epoch {} complete".format(j))
    

#training_data , validation_data , test_data = mnist_loader ,load_data_wrapper ()
#training_data , validation_data , test_data = carga.cargar_estructura_de_datos ()
datos_entrenamiento, datos_validación, datos_prueba = carga.cargar_estructura_de_datos ()

red = Red([784,30,10])
red.SGD(datos_entrenamiento, 1, 10, 3.0, datos_prueba = datos_prueba)



# print(len(list(training_data)))

# # Separating the images from their values
# # x -> input
# # y -> output
# # train_x are the images in the train_set, train_y are the corresponding digits
#     # represented in those images
# train_x, train_y = training_data
# # test_x are the images in the test_set, test_y are the corresponding digits
#     # represented in those images
# test_x, test_y = test_data




#net = Network ([784 , 30, 10])
#net.SGD(training_data , 20, 10, 3.0, test_data = test_data )



# Prueba



import gzip
import pickle

# read the file in read mode as binary 
with gzip.open("C:/Users/USER/Documents/Semestre 10°/Analisis numerico/Analisis-numerico/data/mnist.pkl.gz", 'rb') as file_contents:
    train_set, valid_set, test_set = pickle.load(file_contents, encoding='latin1')

# Separating the images from their values
# x -> input
# y -> output
# train_x are the images in the train_set, train_y are the corresponding digits
    # represented in those images
train_x, train_y = train_set
# test_x are the images in the test_set, test_y are the corresponding digits
    # represented in those images
test_x, test_y = test_set

import matplotlib.cm as cm
import matplotlib.pyplot as plt
# Plotting numpy arrays as images using matplotlib
# show first 5 images in the training set
n = 0
i = 0

while n<numero: 
    respuesta = True
    
    while respuesta == True:
        # plt.subplot(1, 5, i + 1)
        if variable[n][1] == train_y[i]:

            plt.imshow(train_x[i].reshape((28, 28)), cmap=cm.gray_r) # use gray colormap
            plt.axis('off')
            # plt.subplots_adjust(right = 2)
            plt.title('Predicción = %i' % variable[n][0])
            plt.show()

            respuesta = False
        i += 1
    n += 1


