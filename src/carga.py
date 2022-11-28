import pickle as cPickle
import numpy as np
import gzip


#Le y carga los datos de la base de datos mnist. 
#Parametros->  None
#Retorno-> Una tupla de tres arreglos de datos, los cuales cuentan con la informacion de entrenamiento
#          la informacion de validacion y la informacion para realizar las pruebas. 

def cargar_datos():
    datos = gzip.open("./Reconocimiento_Digitos/datos/mnist.pkl.gz", 'rb')
    datos_entrenamiento, datos_validación, datos_prueba = cPickle.load(datos, encoding="latin1")
    datos.close()

    return (datos_entrenamiento, datos_validación, datos_prueba)


#Apartir de los datos cargados crea y modifica las estructuras de forma tal que se puedan trabajar con ellas. 
#Parametros->  None
#Retorno-> Una tupla de tres arreglos de datos, los cuales cuentan con la informacion de entrenamiento
#          la informacion de validacion y la informacion para realizar las pruebas. 
def cargar_estructura_de_datos():

    d_entre, d_vali, d_pru = cargar_datos()

    entradas_entrenamiento =  [np.reshape(x, (784, 1)) for x in d_entre[0]]
    resultados_entrenamiento = [vectorizar(y) for y in d_entre[1]]
    datos_entrenamiento = zip(entradas_entrenamiento, resultados_entrenamiento)

    entradas_validacion =  [np.reshape(x, (784, 1)) for x in d_vali[0]]
    datos_validación = zip(entradas_validacion, d_vali[1])

    entradas_pruebas = [np.reshape(x, (784, 1)) for x in d_pru[0]]
    datos_prueba =  zip(entradas_pruebas, d_pru[1])

    return (datos_entrenamiento, datos_validación, datos_prueba)


#Vectoriza unitariamente un arreglo de 10 pisiciones. 
#Parametros->  posicion: Posicion donde se quiere que el valor sea 1.
#Retorno-> arreglo: Genera un arreglo de 10 arreglos donde en la posicion dada por párametro hay un uno, 
#          y en el resto son ceros. 
def vectorizar(posicion):
    arreglo = np.zeros((10, 1))
    arreglo[posicion] = 1.0

    return arreglo


