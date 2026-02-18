import runpod
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import base64
import io

# Pre-cargamos el modelo fuera del handler para que se mantenga en memoria (Warm Start)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def handler(job):
    # El 'job' contiene la entrada (input) enviada al endpoint
    job_input = job['input']
    img_data = job_input.get('image') # Esperamos un string base64
    
    # Convertir base64 a imagen PIL
    image_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 1. Detectar rostro y extraer el crop
    face = mtcnn(img)
    
    if face is None:
        return {"error": "No face detected"}

    # 2. Generar el embedding (vector)
    # Agregamos una dimensión de batch y pasamos al modelo
    embedding = model(face.unsqueeze(0).to(device)).detach().cpu().tolist()
    
    return {"vector": embedding[0]}

# Arrancamos el servicio
runpod.serverless.start({"handler": handler})
