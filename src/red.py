import numpy as np
import random
import carga
import grafica
import copy


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
        
            
            
        resultados_prueba = [(np.argmax(self.retroalimentacion(x)), y) for (x, y) in datos_prueba]


        return resultados_prueba

    def matriz_de_confusion(self,resultados):
        
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
        for (x, y) in resultados:
            num[y] += 1
            matriz[x][y] +=1
        
        diagonal = 0 
        i = 0
        j = 0 
        
        while i < 10:
            while j < 10:
                
                matriz[i][j] = round(matriz[i][j]/num[j],3)

                if i == j:
                    diagonal += matriz[i][j]

                j += 1
            i += 1
            j = 0

        i = 0
        while i <10:
           print(matriz[i])
           i +=1
        print("")

        return diagonal

    def porcentaje_correctas(self, resultados_prueba):

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

    def SGD_sin_matriz(self, datos_entrenamiento, epocas, mini_lote_tamano, tasa_aprendisaje, datos_prueba=None):

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
                print("Generacion {} : {} / {}".format(j,self.porcentaje_correctas(self.evaluar(datos_prueba)),n_prueba))
                
                
            else:
                print("Generacion {} complete".format(j))

    def SGD_con_matriz(self, datos_entrenamiento, epocas, mini_lote_tamano, tasa_aprendisaje, datos_prueba=None):

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
                print("Generacion {} : {} / {}".format(j,self.porcentaje_correctas(self.evaluar(datos_prueba)),n_prueba))
                if j == epocas - 1:
                    print("la exactitud del modelo es de " + str(round(self.matriz_de_confusion(self.evaluar(datos_prueba)),2))+" siendo 10.0 perfecto")
                
            else:
                print("Generacion {} complete".format(j))

    def predecir(self,elemento_predecir):
        #print(self.retroalimentacion(elemento_predecir))
        retorn = copy.deepcopy(elemento_predecir)
        print("")
        print("La red neuronal predice que el numero que usted ingreso es un " + str(np.argmax(self.retroalimentacion(retorn))))
