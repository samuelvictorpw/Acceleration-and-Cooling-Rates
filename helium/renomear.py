import os

# 1. Use o caminho COMPLETO para o Python não se perder, não importa de onde você rode o script
pasta_imagens = r"C:\Users\Samuel Bernardo\Desktop\FR0catpaper\helium\image\03"

print(f"Buscando arquivos em: {pasta_imagens}")

try:
    # 2. Passa por todos os arquivos dentro da pasta
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

except FileNotFoundError:
    print(f"ERRO: O Python não conseguiu encontrar a pasta exata: {pasta_imagens}")
    print("Verifique se o caminho foi digitado corretamente.")