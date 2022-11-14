import numpy as np
import random
import carga
import grafica
import copy

#Realiza una operacion matematica definica como (1/1+e**-x), en cada osicion del arreglo que entra por parametro.
#Parametros->  parametro: Arreglo con diferentes tamaños que en cada posicion tiene numeros decimales.
#Retorno-> El arreglo que entra por parametro con sus posiciones ya modificadas.
def sigmoidea(parametro):
    
    return 1.0/(1.0+np.exp(-parametro))


#Realiza una operacion matematica definica como (1/1+e**-x)*(1-(1/1+e**-x)), en cada osicion del arreglo que entra por parametro.
#Parametros->  parametro: Arreglo con diferentes tamaños que en cada posicion tiene numeros decimales.
#Retorno-> El arreglo que entra por parametro con sus posiciones ya modificadas.
def primera_sigmoidea(parametro):
    
    return sigmoidea(parametro)*(1-sigmoidea(parametro))


#Se identifica la clase Red, la cual tiene sus funciones propias.
class Red(object):

    #Inicializa los atributos de la clase Red.
    #Parametros->  tamano: Arreglo con diferentes tamaños que en cada posicion tiene un numero entero que representa las neuronas que va tener cada capa.
    #Retorno-> None
    def __init__(self, tamano):

        self.numero_capas = len(tamano)
        self.tamano = tamano
        self.sesgos = [np.random.randn(y, 1) for y in tamano[1:]]
        self.pesos = [np.random.randn(y, x) for x, y in zip(tamano[:-1], tamano[1:])]
        
    #Realiza una multiplicacion y suma matricial entre la entrada con los pesos y los sesgos, en cada iteracion a ese resultado le aplica la sigmoidea.
    #Parametros->  entrada: arrelgo de numeros decimales entre 0 y 1, con el formato 784x1.
    #Retorno->  entrada: retorna la entrada luego de aplicar la operaciones indicadas.
    def retroalimentacion(self,entrada):
        
        for b, w in zip(self.sesgos, self.pesos):
            entrada = sigmoidea(np.dot(w, entrada)+b)
            
        return entrada
        

    #Utilizando un sub-conjunto de datos de entrenamiento realiza actualizaciones sobre los sesgos y los pesos del la clase Red.
    #Parametros->  mini_lote: Sub-conjunto de datos de entrenamiento de un tamaño fijo.
    #              tasa_aprendisaje: Numero decimal que representa la velocidad con la cual quiero que aprenda el modelo.
    #Retorno->  None
    def actualizacion_mini_lote(self, mini_lote, tasa_aprendisaje):

        nabla_b = [np.zeros(b.shape) for b in self.sesgos]
        nabla_w = [np.zeros(w.shape) for w in self.pesos]

        for x, y in mini_lote:
            delta_nabla_b, delta_nabla_w = self.retropropagacion(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.pesos = [w-(tasa_aprendisaje/len(mini_lote))*nw for w, nw in zip(self.pesos, nabla_w)]
        self.sesgos = [b-(tasa_aprendisaje/len(mini_lote))*nb for b, nb in zip(self.sesgos, nabla_b)]


    #Realiza la resta entre dos numeros decimales.
    #Parametros->  salida_activaciones: Arreglo de numeros decimales.
    #              y: Arreglo de numeros decimales.
    #Retorno->  Resta entre los parametrros.
    def costo_derivado(self, salida_activaciones, y):

        return (salida_activaciones-y)


    #Calcula el digito que predice la Red y lo compara con el que deberia dar.
    #Parametros->  datos_prueba: Arreglo con los datos de prueba.
    #Retorno->  resultados_pruebas: Arrelgo de tuplas donde en la primera posicion se tiene el numero predecido y
    #                               en la segunda esta el numero real.
    def evaluar(self,datos_prueba):

        resultados_prueba = [(np.argmax(self.retroalimentacion(x)), y) for (x, y) in datos_prueba]

        return resultados_prueba


    #Genera una matriz de confusion y la imprime en consola.
    #Parametros->  resultados: Arreglo de 10 posiciones, donde en cada posicion indica la correlacion que existen entre la posicion y que el digito dibujado sea
    #                          el numero de la posicion.
    #Retorno->  diagonal: Retorna el la suma de la diagonal de la matriz.
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
        print("")
        while i <10:
           print(matriz[i])
           i +=1
        print("")

        return diagonal


    #Calcula la cantidad de acierto que tuvo la Red, al predecir los digitos.
    #Parametros->  resultados_prueba: Arreglo de tuplas que representan el digito predecido vs el esperado.
    #Retorno->  La cantidad de aciertos que tubo la red.
    def porcentaje_correctas(self, resultados_prueba):

        return sum(int(x == y) for (x, y) in resultados_prueba)


    #Genera una copoa de los sesgos y los pesos, ajustando dichas copias a travez de funciones adicionales. 
    #Parametros->  x: Arreglos de numeros decimales.
    #              y: Arreglos de numeros decimales.
    #Retorno->  Tupla de arreglos.
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


    #Entrena y deja en funcionamiento la red, sin tener que ver las matrices de confusion asociadas al proceso. 
    #Parametros->  datos_entrenamiento: Informacion para realizar el entrenamiento.
    #              epocas: La cantidad de generaciones o iteraciones que va hacer el modelo para entrenarce.
    #              mini_lote_tamano: Numero entero que representa el tamaño de cada mini lote de los datos de entrenamiento.
    #              tasa_aprendisaje: Numero decimal que representa la velocidad con la que va aprender el modelo.
    #              datos_prueba: Arreglos de tuplas con la informacion para hacer pruebas. 
    #Retorno->  None.
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


    #Entrena y deja en funcionamiento la red, viendo las matrices de confusion asociadas al proceso. 
    #Parametros->  datos_entrenamiento: Informacion para realizar el entrenamiento.
    #              epocas: La cantidad de generaciones o iteraciones que va hacer el modelo para entrenarce.
    #              mini_lote_tamano: Numero entero que representa el tamaño de cada mini lote de los datos de entrenamiento.
    #              tasa_aprendisaje: Numero decimal que representa la velocidad con la que va aprender el modelo.
    #              datos_prueba: Arreglos de tuplas con la informacion para hacer pruebas. 
    #Retorno->  None.
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


    #Predice e imprime las probabilidades de que el dibujo sea un digito espesifico. 
    #Parametros->  elemento_predecir: Arrglo de numero decimales entre 0 y 1, con el siguiente formato 784x1.
    #Retorno->  None.
    def predecir(self,elemento_predecir):
        #print(self.retroalimentacion(elemento_predecir))
        retorn = copy.deepcopy(elemento_predecir)
        print("")
        arreglo_relaciones = self.retroalimentacion(retorn)
        prediccion = str(np.argmax(arreglo_relaciones))
        total = (np.sum(arreglo_relaciones))
        
        arreglo_probalidades = []
        arreglo_probalidades_str = []

        for i in arreglo_relaciones:
            arreglo_probalidades.append(np.round(i[0]/total,4))

        for i in arreglo_probalidades:
            i_str = str(i)
            
            if len(i_str) != 6:
                i_str = i_str + "001"
            arreglo_probalidades_str.append(i_str)
        print("La red neuronal predice que el numero que usted ingreso es un ---> " + prediccion+" <---")
        print("")
        print("Y las probabilidades asociado a cada digito son las siguientes:")
        print("-----------------------------------------------------------------------------------------------------")
        
        str_probabilidades = ""
        contador = 0 
        for i in arreglo_probalidades_str:
            str_probabilidades = str_probabilidades + str(contador)+" ---> "+ i + "  | "
            contador += 1
        print(str_probabilidades)