# Reconocimiento_Digitos
El siguiente es un proyecto presentado para la clase de analisis de numerico de la universidad de Los Andes, Colombia. La idea de este, es usar ideas del analisis numerico como lo son las redes neuronales, representadas como tranformaciones lineales multivariadas, y gradiantes estocasticos, como metodo de entrenamiento de dichas redes neuronales, para generar un programa que al ingresar un trazo dibujado de un digito entre 0 y 9, este de como salida, el digito que se encuentra en el trazo. La idea de la costruccion de la red neuronal viene del articulo "Neural networks and deep learning" de  Nielsen, M. Sin embargo la implementacion es propia ademas de nuevas funcionalidades como la posibilidad de dibujar un digito directamente en el programa. 


# Contenido
El proyecto consta de dos carpetas. En la carppeta datos podra encontrar los datos de entrenamiento y prueba necesarios para cargar la red neuronal. Por otro lado, tiene la capeta src la cual contiene cuatro archivos de codigo de python los cual se separan de la siguiente manera. 
        +cargar   -> Se encarga de cargar los datos que se encuentra en la carpeta datos.
        +grafica  -> Se encara de toda la parte de dibujar y mostrar los digitos que entra el usuario.
        +interfaz -> Se encarga de centraliar y como interfas grafica para que el usuario interactue con el programa de forma facil.
        +red      -> Se encarga de administrar y inicializar toda la red neuronal y entrenarla para predecir digitos. 

# Funcionamiento
Para iniciar el programa debe ubicarse en el archivo interzas que se encuentra en la carpeta src del proyecto. Aqui, debera correr el archivo el cual le mostrara el siguiente menu 
        1.   Cargar modelo
        2.   Entrenar
        3.   Generar estadistica
        4.   Predecir digito
        5.   Mostrar digito predecido
        6.   Salir

Antes de hacer cualquier otra cosa el programa le pedira que cargue los datos, es decir que escriba el numero uno (1). Si da cualquier otra opcion le saldra un mensjae donde se le indica que no ha cargado los datos. Al cargar los datos se entrenara por defecto 5 generaciones con 50 mil datos cada una y se hacaran pruebas de efectividad con 10 mil datos, los cuales en cada generacion mostrara cuantos casos fueron correctos sobre el total.

Despues de cargar los datos puede tomar cualquier otra opcion de el menu. Al escribir el numero dos (2), el programa le pedira cuantas generaciones mas quiere entrenar a la red neuronal ya existente. Al escribir el numero tres (3), usted podra generar estadisticas con 100, 1000, 10000 y 50000 datos, dichas estadisticas cosntan de casos favorables sobre casos totales de prueba y una matriz de confusion por cada sub conjutno de datos. Al escribir el numero cuatro (4), se desplegara una venta, en la cual se encuentra un lienzo blanco donde podra dibujar un digito para ser predecido por el programa. Para que ese proceso sea correcto primero debe dibujar el digito, luego dar en el boton terminar  de la ventana emergente y por ultimo cerrar la ventana emergente. Apenbas cierre el programa se mostrara en consola el numero que se predijo y cuanta probabilidad tenia cada digito para ser el dibujado. Al escribir el numero cinco (5), se desplegara una ventana con el dibujo de el digito que se esta prediciendo en ese momento. Al escribir el numero seis (6), el programa terminara su ejecusion. 

# Autores
 Alejandro Caicedo, Camila Gonzalez, Azul Manzano

 Le damos un agradecimiento especial al profesor Mauricio Velasco Pues fue el precursor de este pproyecto y guia importante en el mismo. 