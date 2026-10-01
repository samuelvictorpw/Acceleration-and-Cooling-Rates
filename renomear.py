import os

# Caminho exato da sua pasta de imagens
pasta_imagens = r"C:\Users\Samuel\Desktop\FR0catpaper\image\02"

print("Iniciando a renomeação...")

# Passa por todos os arquivos dentro da pasta
for nome_arquivo in os.listdir(pasta_imagens):
    # Verifica se tem espaço no nome
    if " " in nome_arquivo:
        # Troca o espaço por underline
        novo_nome = nome_arquivo.replace(" ", "_")
        
        # Monta os caminhos completos
        caminho_antigo = os.path.join(pasta_imagens, nome_arquivo)
        caminho_novo = os.path.join(pasta_imagens, novo_nome)
        
        # Renomeia o arquivo
        os.rename(caminho_antigo, caminho_novo)
        print(f"Renomeado: {nome_arquivo}  --->  {novo_nome}")

print("Concluído! Todas as imagens foram renomeadas.")