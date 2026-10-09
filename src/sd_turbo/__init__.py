import torch
from diffusers import AutoPipelineForText2Image

print("Cargando modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32
    # FloatXX 32 o 16
    # Me indica la cantidad de decimales de precision con el que el modelo creara la imagen
    # Si mi pc genera imagen negra, cambiar a 16
)

modelo = modelo.to("cpu")
# cuda - nvidia
# mps - Mac con procesadores MX
# xpu - Grafica intel

prompt = input("Escribe el prompt de la imagen:")
negative_prompt = "Amputee, Bad anatomy, Bad hands, Badly drawn hands, Cloned body, Cloned face, Disgusting proportions, Dismembered, Duplicate, Extra fingers, Extra hands, Extra limbs, Fused fingers, Fused hands, Hideous body, Inadequate scale, Long neck, Malformed limbs, Missing arms, Missing fingers, Missing hands, Missing limbs, Multiple heads, Mutated hands, Mutation, Mutations, Poorly drawn face, Too many fingers, Ugly body, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face",

print("Generando la imagen...")

imagen = modelo(
    prompt=prompt,
    num_interface_steps=10, # Nunumero de veces que creara y
    guidance_scale=2.0, #Que tan fiel sera el prompt
).images[0]

imagen.save("imagen.png")
print("Imagen guardada!")
