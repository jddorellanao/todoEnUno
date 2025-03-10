#En este script se ejemplifican comparaciones de medias de más de dos grupos, 
#bajo esquemas experimentales de todos contra todos y de control contra tratamientos;
#para cada una de ellas se presenta el planteamiento conceptual respectivo y las 
#hipótesis estadísticas correspondentes. Igualmente, se presneta la función y 
#paquetería de R bajo las cuales se implementan de la forma 'función {paquetería}'.

#########################################
#Análisis de varianza
#########################################

#Se quiere conocer si existen diferencias en los resultados de la evaluación de estudiantes 
#sometidos a tres métodos de enseñanza (A: tradicional, B: en línea, C: híbrido). Para esto,
#se tomó una muestra de que contiene la calificación de cinco estudiantes de cada modalidad.

A<-c(78, 85, 82, 88, 90)
B<-c(72, 79, 83, 75, 80)
C<-c(85, 88, 91, 87, 90)
grupos<-factor(c(rep("A",5),rep("B",5),rep("C",5)))

#Se requiere probar si las varianzas son homogéneas (leveneTest {car}).
#H0: S2_A=S2_B=S2_C
#Ha: Por lo menos un grupo tiene varianza diferente

leveneTest(c(A,B,C),group=grupos)

#No hay evidencia de diferencia de varianzas. Así, se require conocer si las muestras 
#se distribuyen de manera normal (shapiro.test {stats}).

#H0: A~N
#Ha: A!~N
shapiro.test(A)

#H0: B~N
#Ha: B!~N
shapiro.test(B)

#H0: C~N
#Ha: C!~N
shapiro.test(C)

#No hay evidencia de que las muestras no se distribuyan de manera normal. Así, dada 
#homogeneidad de varianzas y normalidad de las muestras, se puede realizar análisis
#de varianza (aov {stats}).

#H0: u_A=u_B=u_C
#Ha: Por lo menos un grupo tiene media diferente

anova1<-aov(c(A,B,C)~grupos)
summary(anova1)

#Se rechaza H0; así, es necesaria la prueba post-hoc para conocer cuál o cuáles grupos 
#son diferentes. Dado que la pregunta implica contraste de todos contra todos (tres 
#comparaciones en este caso), se usa prueba de Tukey (TukeyHSD {stats}).
#H0: uA=uB
#Ha: uA!=uB
#H0: uA=uC
#Ha: uA!=uC
#H0: uB=uC
#Ha: uB!=uC

TukeyHSD(anova1)

#Conclusión: Existen diferencias estadísticamente significativas entre los resultados
#de los estudiantes sometidos a los métodos A y B, y entre los de los métodos C y B;
#no hay evidencia de que existan diferencias de resultados entre los estudiantes 
#sometidos a los métodos A y C.

#########################################
#Análisis de varianza. Ejemplo 2
#########################################

#Se quiere conocer el efecto de dos tipos de fertilizante (f1 y f2) sobre le creciemiento
#de las plantas. Con este propósito, Se aplica cada tipo de fertilizante a 12 plantas, 
#mientras que se utiliza un grupo de plantas del mismo tamaño como control (no se les 
#aplica fertilizante). Después de 30 días de la aplicación, se mide el crecimeinto en 
#cm de todos los ejemplares.

control<-c(17.0, 21.0, 15.7, 11.7, 17.2, 15.3, 13.4, 20.7, 13.0, 10.9, 18.9, 17.2)
f1<-c(21.9, 17.0, 18.5, 19.2, 24.6, 23.4, 23.6, 18.3, 17.8, 18.9, 16.8, 16.5)
f2<-c(16.7, 16.8,  9.3, 17.7, 11.4, 14.3, 11.0, 12.1, 18.1, 13.9, 16.3, 16.9)

#Se requiere probar si las varianzas son homogéneas (leveneTest {car})
#H0: S2A=S2B=S2C
#Ha: Por lo menos un grupo tiene varianza diferente

grupos<-factor(c(rep("control",length(control)),rep("f1",length(f1)),rep("f2",length(f2))))
leveneTest(c(control,f1,f2),group=grupos)

#No hay evidencia de diferencia de varianzas. Así, se require conocer si las muestras
#se distribuyen de manera normal (shapiro.test {stats}).

#H0: control~N
#Ha: control!~N
shapiro.test(control)

#H0: f1~N
#Ha: f1!~N
shapiro.test(f1)

#H0: f2~N
#Ha: f2!~N
shapiro.test(f2)

#Dada homogeneidad de varianzas y normalidad de las muestras, se puede realizar análisis
#de varianza (aov {stats}).

#H0: u_control=u_f1=u_f2C
#Ha: Por lo menos un grupo tiene media diferente

anova1<-aov(c(control,f1,f2)~grupos)
summary(anova1)

#Se rechaza H0; así, es necesaria la prueba post-hoc para conocer cuál o cuáles grupos 
#son diferentes. Dado que la pregunta implica contraste de control contra grupos (dos 
#comparaciones en este caso), se usa prueba de Dunnett (DunnettTest {DescTools})

#H0: u_control=u_f1
#Ha: u_control!=u_f1
#H0: u_control=u_f2
#Ha: u_control=u_f2

DunnettTest(c(control,f1,f2)~grupos)

#Conclusión: El fertilizante 1 tiene un efecto significativo sobre el crecimiento de
#las plantas.

#########################################
#Kruskal-Wallis. Ejemplo 1
#########################################

#En una comarca minera se han otorgado tres concesiones de aprovechamiento a tres empresas
#diferentes (El Remanso (er), Minas Regionales (mr) y Grupo García (gg)). Estas empresas
#tienen licencia de operación bajo la premisa de que el suelo de las zonas donde operan 
#no deben tener concentraciones de As por encima de las que son naturales a la región. 
#La línea base regional (lb) se compone de una muestra aleatoria de 10 observaciones 
#realizadas en suelos que no son sometidos a aprovechamiento. Cada año, la autoridad 
#ambiental toma una cantidad de muestras proporcional al tamaño del terreno concesionado
#a cada empresa y es comparado con la línea base. Para el año 2023 se obtuvieron los
#siguientes datos: 

lb<-c(8.5, 13.9, 10.0, 11.4, 9.3, 9.0, 12.3, 12.5, 12.5, 15.0, 8.0, 7.0)
er<-c(12.4, 20.8, 14.1, 8.3, 16.2, 15.0, 16.0, 18.4, 8.0)
mr<-c(13.5, 22.7, 25.0, 19.0)
gg<-c(14.1, 13.8, 15.2, 14.5, 16.0, 15.2)

#Se requiere probar si las varianzas son homogéneas (leveneTest {car}).

#H0: S2lb=S2er=S2mr=S2gg
#Ha: Por lo menos un grupo tiene varianza diferente

grupos<-factor(c(rep("lb",length(lb)),rep("er",length(er)),rep("mr",length(mr)),
  rep("gg",length(gg))))
leveneTest(c(lb,er,mr,gg),group=grupos)

#Por lo menos un grupo tiene varianza dieferente; así, la comparación entre todos
#los grupos (prueba ad-hoc u omnibus) se aborda a través de la prueba de contrastes
#de kruskal-Wallis (kruskal.test {stats}).

#H0: u_lb=u_er=u_mr=u_gg
#Ha: Por lo menos un grupo tiene media diferente

kruskal.test(c(lb,er,mr,gg)~grupos)

#Se rechaza H0; así, es necesaria la prueba post-hoc para conocer cuál o cuáles grupos 
#son diferentes. Dado que la pregunta implica contraste de control contra grupos (tres 
#comparaciones en este caso), se usa prueba de diferencias de Dunnet no paramétrica
#(nparcomp {nparcomp}).

#H0: u_lb=u_er
#Ha: ulb!=u_er
#H0: u_lb=u_mr
#Ha: u_lb!=u_mr
#H0: u_lb=u_gg
#Ha: u_lb!=u_gg

datos<-data.frame(As=c(lb,er,mr,gg),grupos=grupos)
nparcomp(As~grupos,data=datos,t="Dunnett",control="lb")$Analysis

#Conclusión: las concentraciones de As en los terrenos concecionados a Grupo García 
#y del El Remanso son siginifcativamente diferentes a la línea base.

#########################################
#Kruskal-Wallis. Ejemplo 2
#########################################

#Se quiere comparar la edad de formación de las rocas en tres regiones contrastantes
#(región A: cuenca sedimentaria; región B: cordillera montañosa; region C: plataforma de
#carbonatos). Para el efecto, se dataron materiales en muestras de cada una de las regiones
#utilizando U-Th, y se obtuvieron edades en millones de años.

A<-c(320, 350, 330, 410, 340, 550, 380)
B<-c(550, 540, 570, 600, 620, 650, 580)
C<-c(340, 320, 330, 315, 310, 360, 335)

#Se requiere probar si las varianzas son homogéneas (leveneTest {car})
#H0: S2A=S2B=S2C
#Ha: Por lo menos un grupo tiene varianza diferente

grupos<-factor(c(rep("A",length(A)),rep("B",length(B)),rep("C",length(C))))
leveneTest(c(A,B,C),group=grupos)

#No hay evidencia de diferencia de varianzas. Así, se require conocer si las muestras
#se distribuyen de manera normal (shapiro.test {stats}).

#H0: A~N
#Ha: A!~N
shapiro.test(A)

#H0: B~N
#Ha: B!~N
shapiro.test(B)

#H0: C~N
#Ha: C!~N
shapiro.test(C)

#Conclusión: El grupo A no es normal; así, la comparación entre todos los grupos (prueba
#ad-hoc u omnibus) se aborda a través de la prueba de contrastes de kruskal-Wallis
#(kruskal.test {stats})

#H0: uA=uB=uC
#Ha: Por lo menos un grupo tiene media diferente

kruskal.test(c(A,B,C)~grupos)

#Se rechaza H0; así, es necesaria la prueba post-hoc para conocer cuál o cuáles grupos 
#son diferentes. Dado que la pregunta implica contraste de todos contra todos (tres 
#comparaciones en este caso), se usa prueba de diferencias de Tukey no paramétrica
#(nparcomp {nparcomp})

#H0: uA=uB
#Ha: uA!=uB
#H0: uA=uC
#Ha: uA!=uC
#H0: uB=uC
#Ha: uB!=uC

datos<-data.frame(edad=c(A,B,C),grupos=grupos)

nparcomp(edad~grupos,data=datos,t="Tukey")$Analysis

#Las edades de la cuenca sedimentaria y la cordillera montañosa, así como de la 
#cordillera montañosa y la plataforma de carbonatos son estadíticamente diferentes. 
#No existe evidencia que indique diferencias entre las edades de la cuenca sediemnararia
#y la plataforma de carbonatos.