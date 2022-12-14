import matplotlib.cm as cm
import matplotlib.pyplot as plt
import copy

import carga
import grafica
import red

#Genera una instancia de la clase Red y realiza un pequeño entrenamiento de la red neuronal
#Parametros->  datos_entrenamiento: Datos iniciales de entrenamiento
#              datos_validación: Datos inicialies de verificacion
#              datos_prueba: Datos iniciales de prueba
#Retorno-> red_neuronal: Red neuronal con un 90% de presicion. 
def ejecutar_cargar(datos_entrenamiento, datos_validación, datos_prueba):
    red_neuronal = red.Red([784,30,10])
    red_neuronal.SGD_sin_matriz(datos_entrenamiento, 5, 10, 3.0, datos_prueba = datos_prueba)

    return red_neuronal


#Ejecuta un re-entrenamiento con diferentes subconjuntoos de los datos de prueba, los cuales
# tiene difernetes tamaños y genere una matriz de confusion con los resultados obtenidos. 
#Parametros->  datos_entrenamiento: Datos iniciales de entrenamiento
#              datos_validación: Datos inicialies de verificacion
#              datos_prueba: Datos iniciales de prueba
#Retorno-> None
def ejecurar_estadisticas(datos_entrenamiento, datos_validación, datos_prueba)->None:
    red_neuronal = red.Red([784,30,10])
    red_neuronal.SGD_con_matriz(datos_entrenamiento, 5, 10, 3.0, datos_prueba = datos_prueba)
     

#Realiza un entreno adicional sobre una red neuronal ya creada, con unas generaciones espesificas. 
#Parametros->  datos_entrenamiento: Datos iniciales de entrenamiento
#              datos_validación: Datos inicialies de verificacion
#              datos_prueba: Datos iniciales de prueba
#              red: Red neuronal ya creada
#              num_generaciopnes: Numero de generaciones adicionales que quiere entrenar
#Retorno-> None
def ejecutar_entrenar(red,datos_entrenamiento, datos_validación, datos_prueba, num_generaciopnes):
    red.SGD_sin_matriz(datos_entrenamiento, num_generaciopnes, 10, 3.0, datos_prueba = datos_prueba)
    

#Despliega una ventana donde el usuario puede dibujar un digito para que la red neuronal, ya creada, 
# realice una prediccion del digito dibujado. 
#Parametros->  red: Red neuronal ya creada
#Retorno-> imagen: Arreglo de tamaño 784x1 donde cada posicion toma un un pixel del bujo hecho por el 
#                  usuario. 
def ejecutar_predecir(red):
    grafica.main()
    imagen = grafica.retorrnar_imagen()
    red.predecir(imagen)
    
    return imagen


#Despliega una ventana donde el usuario puede visualizar el dibujo hecho anterirmente hecho para que 
# fuera predecido.
#Parametros->  imagen_entrada: Arreglo de tamaño 784x1 donde cada posicion toma un un pixel del bujo 
#                              hecho por el usuario.
#Retorno-> None
def ejecutar_mostrar_imagen(imagen_entrada):
    plt.imshow(imagen_entrada.reshape((28, 28)), cmap=cm.gray_r)  
    plt.axis('off')
    plt.show()


#Muestra el menu del programa.
#Parametros-> None
#Retorno-> None
def mostrar_menu()->None:

    print("-------------------------------------")
    print("              BIENVENIDO              ")
    print("-------------------------------------")
    print("")
    print("Nuestro programa le permite hacer las siguientes opciones")
    print("1.   Cargar modelo")
    print("2.   Entrenar")
    print("3.   Generar estadistica")
    print("4.   Predecir digito")
    print("5.   Mostrar digito predecido")
    print("6.   Salir")
    print("")
    
    
#Inicia el programa para mostrar el menu y poder ejecurar alguna de las opciones del menu.
#Parametros-> None
#Retorno-> None
def ejecutar_programa()->None:

    opcion = 0
    datos_entrenamiento, datos_validación, datos_prueba = carga.cargar_estructura_de_datos ()
    red_neuronal = None
    imagen = []


    while opcion != 6:

        mostrar_menu()
        opcion = int(input("Ingrese el numero de la opcion que desea:    "))


       
        if opcion == 1:
            red_neuronal = ejecutar_cargar(datos_entrenamiento, datos_validación, datos_prueba)
            print("")

        elif opcion == 2:
            if red_neuronal== None:
                print("Por favor carge primero el modelo ")
            else:
                datos_entrenamiento, datos_validación, datos_prueba = carga.cargar_estructura_de_datos ()
                print("")
                num_generaciopnes = int(input("Ingrese el numero de generaciones que desea entrenar: "))
                print("")
                ejecutar_entrenar(red_neuronal,datos_entrenamiento, datos_validación, datos_prueba,num_generaciopnes)

        elif opcion == 3:
            if red_neuronal== None:
                print("Por favor carge primero el modelo ")
            else:
                datos_entrenamiento_100, datos_validación_100, datos_prueba_100 = carga.cargar_estructura_de_datos ()
                datos_entrenamiento_1000, datos_validación_1000, datos_prueba_1000 = carga.cargar_estructura_de_datos()
                datos_entrenamiento_10000, datos_validación_10000, datos_prueba_10000 = carga.cargar_estructura_de_datos()
                datos_entrenamiento_completo, datos_validación_completo, datos_prueba_comppleto = carga.cargar_estructura_de_datos ()

                #Para 100 datos
                print("")
                print("")
                print("Analisis para 100 datos")
                datos_entrenamiento_100 = list(datos_entrenamiento_100)[0:99]
                ejecurar_estadisticas(datos_entrenamiento_100, datos_validación_100, datos_prueba_100)
                print("")
                print("-------------------------------------------------------------------------------")
                print("")
                print("")

                #Para 1000 datos 
                print("Analisis para 1000 datos")
                datos_entrenamiento_1000 = list(datos_entrenamiento_1000)[0:999]
                ejecurar_estadisticas(datos_entrenamiento_1000, datos_validación_1000, datos_prueba_1000)
                print("")
                print("-------------------------------------------------------------------------------")
                print("")
                print("")

                #Para 10000 datos 
                print("Analisis para 10000 datos")
                datos_entrenamiento_10000 = list(datos_entrenamiento_10000)[0:9999]
                ejecurar_estadisticas(datos_entrenamiento_10000, datos_validación_10000, datos_prueba_10000)
                print("")
                print("-------------------------------------------------------------------------------")
                print("")
                print("")

                #Para modelo completo
                print("Analisis para datos completos")
                ejecurar_estadisticas(datos_entrenamiento_completo, datos_validación_completo, datos_prueba_comppleto )
                print("")
                print("-------------------------------------------------------------------------------")
                print("")
                print("")

        elif opcion == 4:
            if red_neuronal== None:
                print("Por favor carge primero el modelo ")
            else:
                
                imagen = ejecutar_predecir(red_neuronal)
                
                
                print("")
        
        elif opcion == 5:
            if red_neuronal== None:
                print("Por favor carge primero el modelo ")
            else:
                if len(list(imagen)) == 0:
                    print("Por favor antes predecir un digito")

                elif len(list(imagen))> 0:
                    ejecutar_mostrar_imagen(imagen)
                    
                    

        elif opcion == 6:
            print("")
            print("Gracias por usar nuestro programa")

        else:
            print("")
            print("Ingrese un valor en el rango establecido")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("")


ejecutar_programa()