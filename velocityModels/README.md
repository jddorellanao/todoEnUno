# Synthetic velocity models
Este repositorio se crea con la finalidad de mostrar y obtener unos de los modelos de velocidades, 2D y 3D, con mayor uso en sísmica de exploración. 

En las carpetas que llevan por título el nombre del modelo se tienen los archivos originales y los archivos en formato `.npy`

# 2D
El cuaderno de trabajo [P_2DSyntheticModelsToNumpy02](https://github.com/jddorellanao/SyntheticVelocityModels/blob/main/P_2DSyntheticModelsToNumpy02.ipynb) consiste en la obtención de tres modelos 2D:
### BP 2004:
Obtenido del sitio web de la [SEG](https://wiki.seg.org/wiki/2004_BP_velocity_estimation_benchmark_model) denominado modelo exacto. Se puede obtener el archivo gz [aqui](http://s3.amazonaws.com/open.source.geoscience/open_data/bpvelanal2004/vel_z6.25m_x12.5m_exact.segy.gz)
### Marmousi2 (Vp):
Modificación del modelo Marmousi. Se puede obtener el modelo en formato `.mat` [aquí](https://drive.google.com/drive/folders/19Sur5hdEB9TpZvmIgBpUqkF_QzGiU96i?usp=sharing). En este caso, se utilizo la versión 'pequeña del modelo'. Estos datos se utilizaron en una inversión sísmica de impedancia acústica, [repo](https://github.com/rafalunelli/SeismicInversion_WGAN-GP)
### SEAM N23900 (Vp):
Obtenido del sitio web de la [SEG](https://wiki.seg.org/wiki/Elastic_2DEW_Classic). El modelo usado es el de la velocidad de la onda P, se puede obtener [aquí](https://drive.google.com/file/d/0B2YKn_VsUkhNalJOWk9naV9MZXc/view?resourcekey=0-4VOHs6uRo0juNxLJThy9Cw).
### SMAART:
Subsalt Multiples Attenuation And Reduction Team, por sus siglas en inglés SMAART. Los modelos Pluto 1.5, Sigsbee2A y Sigsbee2B se encuentran en el sitio web del consorcio, [aquí](http://www.delphi.tudelft.nl/SMAART/)
### Red Sea KFUMP-KAUST:
Modelo del Mar Rojo desarrollado por las universidades KFUMP y KAUST, se puede obtener [aquí](https://wiki.seg.org/wiki/KFUPM-KAUST_Red_Sea_model)
### Ghawar KFUMP:
Modelo del yacimiento Ghawar desarrollado por la universidad KFUMP, se puede leer el paper [aquí](https://link.springer.com/article/10.1007/s12517-019-4390-4) y los datos [aquí](https://www.dropbox.com/sh/cmkg4fnxx2jpxcv/AAB5s0d6rd-wjCPZA3IbmsP-a?dl=0)
### Model94:
Modelo Foothills of the Canadian rockies se puede obtener [aquí](https://reproducibility.org/data/bppublic/PUBLIC_2D_DATASETS/Model94/)

### Correcion
Para evitar la advertencia cuando se obtiene el modelo de velocidades de los archivos SEG-Y, se necesita hacer lo siguiente:

```
# Primera version
vel = np.stack(t.astype(float) for t in s.trace)

# Correcion
vel = np.stack([t.astype(float) for t in s.trace])
```
