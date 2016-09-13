
# coding: utf-8

#  $$ LIBRERIAS$$ $$OPENCV$$

# CONVIRTIENDO ESCALA DE GRISES

# In[1]:

import numpy as np


# In[2]:

import cv2 as cv #Importa la libreria opencv y se renombra


# In[3]:

#Se especifica la ruta donde se encuntra alojada la imagen 

image=cv.imread('images.jpg') #Se carga la imagen a una variable 


# In[4]:

image_gray=cv.cvtColor(image,cv.COLOR_BGR2GRAY)#Se convierte la imagen a la escala de grises


# In[5]:

cv.imwrite('images1.jpg',image_gray)#Se guarda la nueva imagen transformada


# In[6]:

cv.imshow("IMAGEN LOONET TOONS - ORIGINAL", image)#Se muestra la imagen en una ventana


# In[7]:

cv.waitKey()#La imagen permanece en pantalla


# In[8]:

cv.imshow("IMAGEN LOONET TOONS - ESCALA GRISES", image_gray)#Se muestra la imagen en una ventana


# In[9]:

cv.waitKey()#La imagen permanece en pantalla


# OPERACIONES ARIMETICAS CON IMAGENES

# In[10]:

image=cv.imread('images.jpg') #Se carga la imagen a una variable 


# In[11]:

sum_image=(0.3*image+0.1*image)*3 #Se realiza una operacion aritmetica con los valores de la imagen


# In[12]:

invert_image= ~image #se invierten los valores de la imagen original


# In[13]:

rest_image=sum_image - invert_image #Se restan las imagenes  


# In[14]:

cv.imwrite('images2.jpg',rest_image)#Se guarda la nueva imagen 


# In[15]:

cv.imshow("IMAGEN LOONET TOONS - ORIGINAL", image)#Se muestra la imagen en una ventana


# In[16]:

cv.waitKey()#La imagen permanece en pantalla


# In[17]:

cv.imshow("IMAGEN LOONET TOONS - CONTRASTE", rest_image)#Se muestra la imagen en una ventana


# In[18]:

cv.waitKey()#La imagen permanece en pantalla


# VISULAIZAR LOS CAMBIOS DE LA IMAGEN EN TIEMPO REAL

# In[19]:

import cv2 as cv #Importa la libreria opencv y se renombra
import sys


# In[20]:

if len(sys.argv) < 2:
    image = cv.imread('images.jpg') #Se carga la imagen a una variable 
elif len(sys.argv) == 2:
    filename = argv[1]
else:
    sys.exit("hola")


# In[21]:

image_small = cv.resize(image, (800,600))


# In[22]:

textcolor=(240,64,128)


# In[23]:

cv.imshow("IMAGEN LOONET TOONS - IMAGEN AMPLIADA", image_small)#Se muestra la imagen en una ventana


# In[24]:

cv.waitKey()#La imagen permanece en pantalla


# ADICIONAR UN TEXTO A LA IMAGEN EN TIEMPO REAL

# In[148]:

import cv2 as cv #Importa la libreria opencv y se renombra
import sys


# In[149]:

if len(sys.argv) > 2:
    image = cv.imread('imag.jpg') #Se carga la imagen a una variable 
elif len(sys.argv) == 2:
    filename = argv[1]
else:
    sys.exit("hola")


# In[150]:

#Se escirbe un texto en pantalla
cv.putText(image, "Amigos",(50,200),cv.FONT_HERSHEY_PLAIN, 2.5, 0000, thickness=3)


# In[151]:

cv.imshow("IMAGEN LOONET TOONS - CUADRO TEXTO", image)#Se muestra la imagen en una ventana


# In[152]:

cv.waitKey()#La imagen permanece en pantalla


# RECONOCIMEINTO DE ROSTROS Y OJOS POR CAMARA

# In[1]:

import numpy as np
import cv2
import sys


# In[2]:

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


# In[3]:

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[ ]:

cap = cv2.VideoCapture(0)


# In[ ]:

while(True):
    # Captura de fotograma
    ret, frame = cap.read()

    # Operaciones
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    #Dibujamos un rectangulo en las coordenadas de cada rostro
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color =frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
    # Mostrar la imagen que resulta
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# In[ ]:

# cunado se realiza todo se libera la captura
cap.release()
cv2.destroyAllWindows()


# RECONOCIMEINTO DE ROSTROS Y OJOS EN UNA IMAGEN

# In[ ]:

import numpy as np
import cv2
import sys


# In[ ]:

#cargamos la plantilla e inicializamos la webcam:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


# In[ ]:

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[ ]:

image = cv2.imread('megan.jpg') #Se carga la imagen a una variable 


# In[ ]:

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# In[ ]:

faces = face_cascade.detectMultiScale(gray, 1.3, 5)


# In[ ]:

#Dibujamos un rectangulo en las coordenadas de cada rostro
   
for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color =image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow('image',image)
        cv2.waitKey()#La imagen permanece en pantalla
        cv2.destroyAllWindows()


# DETECTAR COLOR VERDE  

# In[7]:

import cv2
import numpy as np

#Iniciamos la camara
captura = cv2.VideoCapture(0)
 
while(1):
     
    #Capturamos una imagen y la convertimos de RGB -> HSV
    _, imagen = captura.read()
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
 
    #Establecemos el rango de colores que vamos a detectar
    #En este caso de verde oscuro a verde-azulado claro
    verde_bajos = np.array([49,50,50], dtype=np.uint8)
    verde_altos = np.array([80, 255, 255], dtype=np.uint8)
 
    #Crear una mascara con solo los pixeles dentro del rango de verdes
    mask = cv2.inRange(hsv, verde_bajos, verde_altos)
 
    #Encontrar el area de los objetos que detecta la camara
    moments = cv2.moments(mask)
    area = moments['m00']
 
    #Descomentar para ver el area por pantalla
    print area
    if(area > 2000000):
         
        #Buscamos el centro x, y del objeto
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])
         
        #Mostramos sus coordenadas por pantalla
        print "x = ", x
        print "y = ", y
 
        #Dibujamos una marca en el centro del objeto
        cv2.rectangle(imagen, (x-5, y-5), (x+5, y+5),(0,0,255), 2)
        cv2.putText(imagen, "pos:"+ str(x)+","+str(y), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
     
    #Mostramos la imagen original con la marca del centro y
    #la mascara
    cv2.imshow('mask', mask)
    cv2.imshow('Camara', imagen)
    
    #tecla = cv2.waitKey(5) & 0xFF
    if cv2.waitKey(1) & 0xFF == ord('q'):
        captura.release()
        break
 
cv2.destroyAllWindows()






# DETECCION DE BORDES

# In[10]:

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('megan.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


# DETECTOR DE ESQUINAS

# In[5]:

import cv2
import numpy as np

filename = 'images.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#resultado se dilata para marcar las esquinas
dst = cv2.dilate(dst,None)

# Umbral para un valor óptimo, que puede variar dependiendo de la imagen.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('Imagen',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# ESCALA DE GRISES EN TIEMPO REAL

# In[2]:

import cv2
import numpy as np
cap=cv2.VideoCapture(0)
while (True):
    ret, frame=cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# RECONOCIMIENTO DE FORMAS

# In[ ]:

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, img = cap.read()
    cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
    crop_img = img[100:300, 100:300]
    grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    value = (37,37) 
    blurred = cv2.GaussianBlur(grey, value, 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255,
                               cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
   
    _, contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    max_area = -1
    print len(contours)
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnt=contours[ci]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(crop_img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
    cv2.drawContours(drawing,[hull],0,(0,0,255),0)
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img,far,1,[0,0,255],-1)
        dist = cv2.pointPolygonTest(cnt,far,True)
        cv2.line(crop_img,start,end,[0,255,0],2)
        cv2.circle(crop_img,far,5,[0,0,255],-1)

    if count_defects>3:
        cv2.putText(img,"ABIERTA", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 4)
    else:
        cv2.putText(img,"CERRADA", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 4)

    cv2.imshow('drawing', drawing)
    cv2.imshow('end', crop_img)
    cv2.imshow('Gesture', img)

    all_img = np.hstack((drawing, crop_img))
    cv2.imshow('Contours', all_img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        break


# In[ ]:




# In[ ]:




# In[ ]:



