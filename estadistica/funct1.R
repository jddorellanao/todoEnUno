test_final <- function(datos) {
    require(car)
    require(DescTools)
    library(nparcomp)
    controlA(datos)
    nmuestras <- p_nmuestras()
    controlB(datos,nmuestras)
}

# ----------------------------------------------------
# verificar que tenga el formato correcto
controlA <- function(datos) {
    control1(datos)
    control2(datos)
    control3(datos)
}

# verificar si es un data.frame
control1 <- function(datos) { 
    if (!is.data.frame(datos)) {
        imp_ejem()
        stop("El input debe ser un dataframe.")
    }
}

# verificar que tenga dos o mas columnas
control2 <- function(datos) {
    if (ncol(datos) < 2) {
        imp_ejem()
        stop(paste("El dataframe no tiene el numero de columnas requeridas por la función", ncol(datos)))
    }
}

# verificar que una columna sea clase factor
control3 <- function(datos) {    
    es_factor <- sapply(datos, is.factor)
    if (!any(es_factor)) {
        imp_ejem()
        stop("El dataframe debe contener al menos una columna de tipo factor.")
    }
}

# ----------------------------------------------------
# imprimir ejemplo de formato
imp_ejem <- function() {
    print("El formato correcto tiene que ser similiar a:")
    cat("Observaciones 1 | ... | Factor\n")
    cat("----------------|-----|--------\n")
    cat("       1        | ... | GrupoA \n")
    cat("      ...       | ... |   ...  \n")
    cat("       1        | ... | GrupoZ \n")
}

# ----------------------------------------------------
# cantidad de muestras a evaluar
p_nmuestras <- function() {
    nmuestras <- readline(prompt = "¿Cuántas muestras vas a evaluar? (ingrese al número correspondiente)\n1. Una\n2. Dos\n3. Dos o más\n")
    return(as.integer(nmuestras))
}

#-----------------------------------------------------
# verificar valores ingresados numericos
ver_num <- function(prompt) {
    valor <- readline(prompt = prompt)

    # Intentar convertir el valor a numérico
    valor_n <- as.integer(valor)

    # Verificar si la conversión fue exitosa
    if (is.na(valor_n)) {
        stop("El valor ingresado no es un número válido. Por favor ingrese un número.")
    }
    return(valor_n)
}

# ----------------------------------------------------
# verificar que se cumpla para el caso de n-muestras
controlB <- function(datos, nmuestras) {
    if (nmuestras==1){
        control4(datos)
    } else if (nmuestras==2){
        control5(datos)
    } else if (nmuestras==3){
        control6(datos)
    } else {
        stop("Valor ingresado incorrecto")
    }
}

#------------------------------------------------------------------
# verificar que se cumpla para 1 muestra
control4 <- function(datos) {
    # seleccionar columna factor
    col_f <- names(datos)[sapply(datos, is.factor)]
    cat("Columnas de tipo factor disponibles:", paste(col_f, collapse = ", "), "\n")
    colf <- readline(prompt = "Selecciona la columna de tipo factor (ingrese el nombre correctamente): \n")
    # Verificar que la selección del usuario es válida
    if (!(colf %in% col_f)) {
        stop("La columna ingresada no es válida. Por favor ingrese un nombre de columna correcto.")
    }

    # seleccionar categoria
    lvls <- levels(datos[[colf]])
    cat("Categoría(s):\n")
    print(lvls)
    cat("\n")
    categoria <- readline(prompt = "¿Cuál es tu categoría? (ingrese el nombre correctamente)")
    # Verificar que la selección del usuario es válida
    if (!(categoria %in% lvls)) {
        stop("La categoría ingresada no existe en los niveles disponibles.")
    }

    # seleccionar observaciones
    col_nf <- names(datos)[!sapply(datos, is.factor)]
    cat("Columna(s) de observación(es):", paste(col_nf, collapse = ", "), "\n")
    obs <- readline(prompt = "¿Cuál columna de observaciones? (ingrese el nombre correctamente):\n")
    # Verificar que la columna ingresada existe en las columnas no factor
    if (!(obs %in% col_nf)) {
        cat("La columna ingresada no es válida. Por favor ingrese un nombre de columna correcto.\n")
    }

    alternative <- readline(prompt = "¿Cuál es la HA?: (ingrese número correspondiente)\n1. Mayor que\n2. Menor que\n3. Igual\n")
    if(alternative==1){
        alt="greater"
    } else if (alternative==2) {
        alt="greater"
    } else if (alternative==3) {
        alt="two.sided"
    } else {
        stop("Opción incorrecta")
    }
    alpha <- ver_num("¿Cuál es el valor de α?: ")
    mu <- ver_num("¿Cuál es el valor de μ?: ")
    obs_fil <- datos[datos[[colf]] == categoria, obs]

    pval <- shapiro.test(obs_fil)$p.value
    cat("Resultados del p-value del test de Shapiro-Wilk:\n")
    print(pval)

    if(pval>alpha){
        cat("No hay evidencia que sugiera que los datos no se distribuyen de manera normal,\n")
        cat("se procede con la prueba T-student")
        t.test(obs_fil, mu=mu, alternative=alt)

    }else if (pval<alpha) {
        cat("Los datos no se distrubyen de manera normal,\n")
        cat("se procede con la prueba Wilcoxon")
        wilcox.test(obs_fil, mu=mu, alternative=alt)
    }
}   

#------------------------------------------------------------------
# verificar que se cumpla para 2 muestra
control5 <- function(datos) {
    # seleccionar la columna factor
    col_f <- names(datos)[sapply(datos, is.factor)]
    cat("Columnas de tipo factor disponibles:", paste(col_f, collapse = ", "), "\n")
    colf <- readline(prompt = "Selecciona la columna de tipo factor (ingrese el nombre correctamente): \n")
    # Verificar que la selección del usuario es válida
    if (!(colf %in% col_f)) {
        stop("La columna ingresada no es válida. Por favor ingrese un nombre de columna correcto.")
    }

    # seleccionar las categorias a analizar
    lvls <- levels(datos[[colf]])
    if (length(lvls) < 2) {
        stop("La columna seleccionada debe tener al menos 2 categorías.")
    } else {
        cat("Las categorías disponibles son:\n")
        for (i in seq_along(lvls)) {
            cat(i, ":", lvls[i], "\n")
        }
    }

    # Solicitar al usuario que seleccione dos categorías diferentes
    sel1<-readline(prompt = "Seleccione la 1era categoría: ")
    sel2<-readline(prompt = "Seleccione la 2nda categoría: ")
    if (sel1 == sel2) {
        stop("Debe seleccionar dos categorías diferentes.")
    }
    
    #---------------------------------------
    # pareadas o no
    paired<-readline(prompt = "¿Están asociadas? (ingrese al número correspondiente)\n1. Sí\n2. No\n")

    if (paired==1) {
        paired=TRUE
    }else if (paired==2) {
        paired=FALSE
    }else {
        stop("Error al definir la asociación de las muestras")
    }

    # seleccionar observaciones
    col_nf <- names(datos)[!sapply(datos, is.factor)]
    cat("Columna(s) de observación(es):", paste(col_nf, collapse = ", "), "\n")
    obs <- readline(prompt = "¿Cuál columna de observaciones? (ingrese el nombre correctamente):\n")
    # Verificar que la columna ingresada existe en las columnas no factor
    if (!(obs %in% col_nf)) {
        cat("La columna ingresada no es válida. Por favor ingrese un nombre de columna correcto.\n")
    }
    obs<-as.character(obs)
    sel1<-as.character(sel1)
    sel2<-as.character(sel2)
    alternative <- readline(prompt = "¿Cuál es la HA?: (ingrese número correspondiente)\n1. Mayor que\n2. Menor que\n3. Igual\n")
    if(alternative==1){
        alt="greater"
    } else if (alternative==2) {
        alt="greater"
    } else if (alternative==3) {
        alt="two.sided"
    } else {
        stop("Opción incorrecta")
    }
    mu <- ver_num("¿Cuál es el valor de μ?: ")
    alpha <- ver_num("¿Cuál es el valor de α?: ")
    iris[iris[['Species']] == 'versicolor', 'Sepal.Length']
    obs_fil1 <- datos[datos[[colf]] == sel1, obs]
    obs_fil2 <- datos[datos[[colf]] == sel2, obs]
    #-----------------------------------
    # pareados
    if (paired) {
        d <- obs_fil1 - obs_fil2
        pval <- shapiro.test(d)$p.value
        cat("Resultados del p-value del test de Shapiro-Wilk:\n")
        print(pval)
        #-----------------------------------------
        # prueba de normalidad
        if(pval>alpha){
            cat("No hay evidencia que sugiera que los datos no se distribuyen de manera normal,\n")
            cat("se procede con la prueba T-student")
            t.test(d, mu=mu, alternative=alt)

        }else if (pval<alpha) {
            cat("Los datos no se distrubyen de manera normal,\n")
            cat("se procede con la prueba Wilcoxon")
            wilcox.test(d, mu=mu, alternative=alt)
        }
    } else {
        pval1 <- shapiro.test(obs_fil1)$p.value
        pval2 <- shapiro.test(obs_fil2)$p.value
        cat("Resultados del p-value del test de Shapiro-Wilk:\n")
        cat(paste(pval1, pval2, sep = " "), "\n")
        #-----------------------------------------
        # prueba de normalidad
        if (pval1>alpha & pval2>alpha) {
            cat("No hay evidencia que sugiera que los datos no se distribuyen de manera normal,\n")
            cat("se procede con la prueba Levene\n")
            grupos<-factor(c(rep(sel1,length(obs_fil1)),rep(sel2,length(obs_fil2))))
            pvall <- leveneTest(c(obs_fil1,obs_fil2),group=grupos)$`Pr(>F)`[1]
            print(pvall)
        #-----------------------------------------
        # prueba de varianzas homogeneas
            if (pvall>alpha) {
                cat("Dada homogeneidad de varianzas,\n")
                cat("se procede con la prueba T-student con varianza agrupada\n")
                t.test(obs_fil1, obs_fil2, mu=mu, alternative=alt, var.equal=T)
            } else if (pvall<alpha) {
                cat("No hay evidencia que las varianzas sean homogeneas,\n")
                cat("se procede con la prueba T-student con varianza separada\n")
                t.test(obs_fil1, obs_fil2, mu=mu, alternative=alt, var.equal=F)
            }

        } else {
            cat("Los datos no se distribuyen de manera normal,\n")
            cat("se procede con la prueba de Mann-Whitney")
            wilcox.test(obs_fil1, obs_fil2, mu=mu, alternative=alt)
        }
    }
}

# verificar que se cumpla para 2+ muestra
control6 <- function(datos) {
    # seleccionar la columna factor
    col_f <- names(datos)[sapply(datos, is.factor)]
    cat("Columnas de tipo factor disponibles:", paste(col_f, collapse = ", "), "\n")
    colf <- readline(prompt = "Selecciona la columna de tipo factor (ingrese el nombre correctamente): \n")
    # Verificar que la selección del usuario es válida
    if (!(colf %in% col_f)) {
        stop("La columna ingresada no es válida. Por favor ingrese un nombre de columna correcto.")
    }
    lvls <- levels(datos[[colf]])
    if (length(lvls) < 2) {
        stop("La columna seleccionada debe tener al menos 2 categorías.")
    } else {
        cat("La columna", colf, "tiene", length(lvls), "categorías:\n")
        print(lvls)
        cat("Se utilizarán todas.\n") 
    }
    # seleccionar observaciones
    col_nf <- names(datos)[!sapply(datos, is.factor)]
    cat("Columna(s) de observación(es):", paste(col_nf, collapse = ", "), "\n")
    obs <- readline(prompt = "¿Cuál columna de observaciones? (ingrese el nombre correctamente):\n")
    # Verificar que la columna ingresada existe en las columnas no factor
    if (!(obs %in% col_nf)) {
        cat("La columna ingresada no es válida. Por favor ingrese un nombre de columna correcto.\n")
    }

    alpha <- ver_num("¿Cuál es el valor de α?: ")

    #-------------------------------
    # Levene test
    pval<-leveneTest(datos[[obs]] ~ datos[[colf]])$`Pr(>F)`[1]
    cat("p value de la prueba Levene\n")
    print(pval)
    if (pval>alpha) { # levenetest si
        cat("No hay evidencia de diferencia de varianzas\n")
        resumen <- summary(aov(datos[[obs]] ~ datos[[colf]]))

        #-------------------------------
        # evaluamos shapiro
        shapiro_rs <- list()
        for (categoria in lvls) {
            obs_fil <- datos[datos[[colf]] == categoria, obs]
            shapiro_rs[[categoria]] <- shapiro.test(obs_fil)$p.value
        }

        # verificar 1x1 los valores de shapiro
        shap_ <- TRUE
        for (categoria in names(shapiro_rs)) {
            valor_p <- shapiro_rs[[categoria]]
            if (valor_p < alpha) {
                shap_ <- FALSE
                break
            }
        }
        cat("Valores de pvalue de Shapiro:", paste(shap_, collapse = ", "), "\n")

        # ----------------------------
        # analisis de varianzas
        if (shap_) { # shapirotest si
            cat("Los grupos son normales.\n")
            anova1<-aov(datos[[obs]] ~ datos[[colf]])
            resumen <- summary(anova1)
            pval <- resumen[[1]]$`Pr(>F)`[1]
            cat("Valor de pvalue de ANOVA:", paste(pval, collapse = ", "), "\n")

            if (pval<alpha) { # aov si
                cat("Por lo menos un grupo tiene media diferente.\n")
                # ----------------------------
                # Contrasates
                # preguntar si es TvT o CvG
                tod <- as.integer(readline(prompt = "¿Tienes grupo de control?: (ingrese número correspondiente)\n1. Sí\n2. No\n"))
                if (tod==1) {
                    # CvG
                    cat("Grupos disponibles:", paste(lvls, collapse = ", "), "\n")
                    tod_ <- readline(prompt = "¿Cuál es tu grupo de control?.\n")
                    DunnettTest(x=datos[[obs]], g=datos[[colf]], control=tod_)
                } else if (tod==2) {
                    # TvT
                    TukeyHSD(anova1)
                }
            } else if(pval>alpha) { # aov no
                cat("No hay evidencia de diferencia entre grupos.\n")
            }
        } else { # shapirotest no
            cat("Los grupos no son normales.\n")
            pval <- kruskal.test(datos[[obs]]~datos[[colf]])$p.value
            cat("Valor de pvalue de Kruskal-Wallis:", paste(pval, collapse = ", "), "\n")
            As<-datos[[obs]]
            gruppos<-datos[[colf]]
            if (pval<alpha) { # kruskal si
                # ----------------------------
                # Contrasates
                # preguntar si es TvT o CvG
                tod <- as.integer(readline(prompt = "¿Tienes grupo de control?: (ingrese número correspondiente)\n1. Sí\n2. No\n"))
                if (tod==1) {
                    # CvG
                    cat("Grupos disponibles:", paste(lvls, collapse = ", "), "\n")
                    tod_ <- readline(prompt = "¿Cuál es tu grupo de control?.\n")
                    nparcomp(As~gruppos, data=datos, t="Dunnett", control=as.character(tod_))$Analysis
                } else if (tod==2) {
                    # TvT
                    nparcomp(As~gruppos, data=datos, t="Tukey")$Analysis
                }
            } else if(pval>alpha) { # kruskal no
                cat("No hay evidencia de diferencia entre grupos.\n")
            }

        }
    } else { # levenetest no
        cat("Las varianzas son diferentes\n")
        pval <- kruskal.test(datos[[obs]]~datos[[colf]])$p.value
        cat("Valor de pvalue de Kruskal-Wallis:", paste(pval, collapse = ", "), "\n")
        if (pval<alpha) {
            # ----------------------------
            # Contrasates
            As<-datos[[obs]]
            gruppos<-datos[[colf]]
            # preguntar si es TvT o CvG
            tod <- as.integer(readline(prompt = "¿Tienes grupo de control?: (ingrese número correspondiente)\n1. Sí\n2. No\n"))
            if (tod==1) {
                # CvG
                cat("Grupos disponibles:", paste(lvls, collapse = ", "), "\n")
                tod_ <- readline(prompt = "¿Cuál es tu grupo de control?.\n")
                nparcomp(As~gruppos, data=datos, t="Dunnett", control=as.character(tod_))$Analysis
            } else if (tod==2) {
                # TvT
                nparcomp(As~gruppos, data=datos, t="Tukey")$Analysis
            }
        } else if(pval>alpha) {
            cat("No hay evidencia de diferencia entre grupos.\n")
        }
    }
}