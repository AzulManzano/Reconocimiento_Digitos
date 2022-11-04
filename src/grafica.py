import sys
from tkinter import*
from turtle import width
from PIL import Image, ImageDraw
import numpy as np
import copy


deawing_area = ""
x,y= None,None
count = 0 
image_coun = 0

image_name = "canvas_"
pil_image =Image.new("1", (280,280),"white")
draw = ImageDraw.Draw(pil_image)

imagen = None

def retorrnar_imagen():
    global deawing_area,pil_image,draw,imagen
    deawing_area = ""
    pil_image =Image.new("1", (280,280),"white")
    draw = ImageDraw.Draw(pil_image)
    retorn = copy.deepcopy(imagen)
    imagen = []
    
    return retorn

def quitar(event):
    sys.exit()

def limpiar(event):
    global deawing_area,pil_image,draw
    deawing_area.delete("all")
    pil_image =Image.new("1", (280,280),"white")
    draw = ImageDraw.Draw(pil_image)

def graficar(event):
    global deawing_area,x,y,count,image_coun
    newx, newy = event.x, event.y

    if x is None:
        x,y = newx,newy
        return 
    count += 1
    #sys.stdout.write("revent count %d" % count)
    deawing_area.create_line((x,y,newx,newy),width = 10, smooth = True)
    draw.line((x,y,newx,newy),width = 10)
    x,y =newx,newy

def graficar_finalizo(event):
    global x,y 
    x,y = None,None

def guardar(event):
    global pil_image, image_name,image_coun, imagen
    image_coun +=1
    file_name = image_name + str(image_coun)+ ".jpg"
    pil_image = pil_image.resize((28,28), Image.ANTIALIAS)
    imagen_arreglo = np.asanyarray(pil_image)
    imagen_arreglo_unitario = np.reshape(imagen_arreglo, (784, 1))
    imagen_arreglo_unitario_numerico = []

    for i in range(0,len(imagen_arreglo_unitario)):
        if imagen_arreglo_unitario[i] == False:
            imagen_arreglo_unitario_numerico.append([1])
        elif imagen_arreglo_unitario[i] == True:
            imagen_arreglo_unitario_numerico.append([0])
    np_array = np.array(imagen_arreglo_unitario_numerico).reshape(784, 1)
    imagen = np_array 


def main():
    global deawing_area
    win = Tk()
    win.title("Digito")
    deawing_area = Canvas(win, width=280, height = 280, bg="white")
    deawing_area.bind("<B1-Motion>",graficar)
    deawing_area.bind("<ButtonRelease-1>",graficar_finalizo)
    deawing_area.pack()


    # b1 = Button(win,text="Salir",bg="CadetBlue")
    # b1.pack()
    # b1.bind("<Button-1>",quitar)

    b2 = Button(win,text="Limpiar",bg="CadetBlue")
    b2.pack()
    b2.bind("<Button-1>",limpiar)

    b3 = Button(win,text="Terminar",bg="CadetBlue")
    b3.pack()
    b3.bind("<Button-1>",guardar)
    
    win.mainloop()
    
