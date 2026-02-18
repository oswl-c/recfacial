FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# Instalar dependencias
RUN pip install runpod facenet-pytorch pillow

# Copiar tu script
COPY handler.py /handler.py

# Comando para ejecutar el handler al iniciar el pod
CMD [ "python", "-u", "/handler.py" ]
