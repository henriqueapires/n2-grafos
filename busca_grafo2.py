import time
import matplotlib.pyplot as plt
import networkx as nx

rede_social = {
    'A': [('B', 4), ('C', 1), ('D', 7), ('Bot1', 8)],  # Bot1 conectado diretamente a A
    'B': [('A', 4), ('E', 3), ('F', 6), ('Bot2', 10)],  # Bot2 conectado diretamente a B
    'C': [('A', 1), ('D', 2), ('G', 5)],  
    'D': [('A', 7), ('C', 2), ('E', 3), ('H', 4), ('Bot3', 9)],  # Bot3 conectado a D
    'E': [('B', 3), ('D', 3), ('F', 2), ('I', 6)],  
    'F': [('B', 6), ('E', 2), ('J', 3), ('Bot4', 11)],  # Bot4 conectado a F
    'G': [('C', 5), ('H', 2), ('J', 4)],  
    'H': [('D', 4), ('G', 2), ('I', 1)],  
    'I': [('E', 6), ('H', 1), ('J', 5)],  
    'J': [('F', 3), ('G', 4), ('I', 5), ('Bot5', 13)],  # Bot5 conectado a J
    'Bot1': [('A', 8), ('Bot6', 12)],  # Bot1 conectado à rede principal e outro bot
    'Bot2': [('B', 10), ('Bot7', 14)],  # Bot2 conectado à rede principal e outro bot
    'Bot3': [('D', 9), ('Bot8', 15)],  # Bot3 conectado à rede principal e outro bot
    'Bot4': [('F', 11), ('Bot6', 10)],  # Bot4 conectado à rede principal e outro bot
    'Bot5': [('J', 13), ('Bot7', 12)],  # Bot5 conectado à rede principal e outro bot
    'Bot6': [('Bot1', 12), ('Bot4', 10)],  # Bot6 conectado entre bots
    'Bot7': [('Bot2', 14), ('Bot5', 12)],  # Bot7 conectado entre bots
    'Bot8': [('Bot3', 15)],  # Bot8 apenas conectado a Bot3
}

# rede_social = {
#     'A': [('B', 4), ('C', 1), ('D', 7), ('Bot1', 8)],  # Bot1 conectado diretamente a A
#     'B': [('A', 4), ('E', 3), ('F', 6), ('Bot2', 10)],  # Bot2 conectado diretamente a B
#     'C': [('A', 1), ('D', 2), ('G', 5)],  
#     'D': [('A', 7), ('C', 2), ('E', 3), ('H', -4), ('Bot3', 9)],  # Bot3 conectado a D e aresta negativa para H
#     'E': [('B', 3), ('D', 3), ('F', -2), ('I', 6)],  # Peso negativo entre E e F
#     'F': [('B', 6), ('E', -2), ('J', 3), ('Bot4', 11)],  # Bot4 conectado a F
#     'G': [('C', 5), ('H', 2), ('J', 4)],  
#     'H': [('D', -4), ('G', 2), ('I', 1)],  
#     'I': [('E', 6), ('H', 1), ('J', -5)],  # Peso negativo entre I e J
#     'J': [('F', 3), ('G', 4), ('I', -5), ('Bot5', 13)],  # Bot5 conectado a J
#     'Bot1': [('A', 8), ('Bot6', 12)],  # Bot1 conectado à rede principal e outro bot
#     'Bot2': [('B', 10), ('Bot7', 14)],  # Bot2 conectado à rede principal e outro bot
#     'Bot3': [('D', 9), ('Bot8', 15)],  # Bot3 conectado à rede principal e outro bot
#     'Bot4': [('F', 11), ('Bot6', 10)],  # Bot4 conectado à rede principal e outro bot
#     'Bot5': [('J', 13), ('Bot7', 12)],  # Bot5 conectado à rede principal e outro bot
#     'Bot6': [('Bot1', 12), ('Bot4', 10)],  # Bot6 conectado entre bots
#     'Bot7': [('Bot2', 14), ('Bot5', 12)],  # Bot7 conectado entre bots
#     'Bot8': [('Bot3', 15)],  # Bot8 apenas conectado a Bot3
# }

usuarios = list(rede_social.keys())

def visualizar_grafo(rede_social, possiveis_bots, origem, destino):
    G = nx.DiGraph()
    
    for usuario, conexoes in rede_social.items():
        for vizinho, peso in conexoes:
            G.add_edge(usuario, vizinho, weight=peso)
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 10))
    
    node_colors = []
    for node in G.nodes():
        if node == origem:
            node_colors.append("green")       # origem em verde
        elif node == destino:
            node_colors.append("orange")      # destino em laranja
        elif node in possiveis_bots:
            node_colors.append("red")         # possíveis bots em vermelho
        else:
            node_colors.append("skyblue")     # demais nós em azul claro
    
    nx.draw_networkx(G, pos, with_labels=True, node_size=400, node_color=node_colors, font_size=8, font_weight="bold", arrows=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Rede Social - Representação Visual do Grafo")
    plt.show()

# função de busca em profundidade (DFS)
def dfs(rede_social, origem, destino):
    start_time = time.time()
    caminhos = []
    
    def dfs_recursive(usuario_atual, caminho):
        if usuario_atual == destino:
            caminhos.append(caminho)
            return
        for (vizinho, _) in rede_social.get(usuario_atual, []):
            if vizinho not in caminho:
                dfs_recursive(vizinho, caminho + [vizinho])
                
    dfs_recursive(origem, [origem])
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    if not caminhos:
        print(f"Não existe caminho de {origem} para {destino} usando DFS.")
    else:
        print(f"Caminhos encontrados usando DFS de {origem} para {destino}:")
        for caminho in caminhos:
            print(" -> ".join(caminho))
        caminho_otimo = min(caminhos, key=len)
        print(f"Caminho ótimo: {' -> '.join(caminho_otimo)}")
    print(f"Tempo de processamento do DFS: {tempo_processamento:.6f} segundos\n")
    return caminhos

# função de busca em largura (BFS)
def bfs(rede_social, origem, destino):
    start_time = time.time()
    caminhos = []
    from collections import deque
    
    fila = deque()
    fila.append((origem, [origem]))
    
    while fila:
        (usuario, caminho) = fila.popleft()
        for (vizinho, _) in rede_social.get(usuario, []):
            if vizinho not in caminho:
                novo_caminho = caminho + [vizinho]
                if vizinho == destino:
                    caminhos.append(novo_caminho)
                else:
                    fila.append((vizinho, novo_caminho))
                    
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    if not caminhos:
        print(f"Não existe caminho de {origem} para {destino} usando BFS.")
    else:
        print(f"Caminhos encontrados usando BFS de {origem} para {destino}:")
        for caminho in caminhos:
            print(" -> ".join(caminho))
        caminho_otimo = min(caminhos, key=len)
        print(f"Caminho ótimo: {' -> '.join(caminho_otimo)}")
    print(f"Tempo de processamento do BFS: {tempo_processamento:.6f} segundos\n")
    return caminhos

# função do algoritmo de dijkstra
def dijkstra(rede_social, origem, destino):
    start_time = time.time()
    for arestas in rede_social.values():
        for (_, peso) in arestas:
            if peso < 0:
                print("A rede social contém interações com pesos negativos. O algoritmo de Dijkstra não pode ser aplicado.")
                return []

    import heapq

    fila = []
    heapq.heappush(fila, (0, origem, [origem]))
    visitados = set()

    while fila:
        (custo, usuario, caminho) = heapq.heappop(fila)
        if usuario in visitados:
            continue
        visitados.add(usuario)

        if usuario == destino:
            end_time = time.time()
            tempo_processamento = end_time - start_time
            print(f"Caminho ótimo encontrado pelo algoritmo de Dijkstra de {origem} para {destino}:")
            print(" -> ".join(caminho))
            print(f"Custo total: {custo}")
            print(f"Tempo de processamento do Dijkstra: {tempo_processamento:.6f} segundos\n")
            return [caminho]

        for (vizinho, peso) in rede_social.get(usuario, []):
            if vizinho not in visitados:
                heapq.heappush(fila, (custo + peso, vizinho, caminho + [vizinho]))

    end_time = time.time()
    tempo_processamento = end_time - start_time
    print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Dijkstra.")
    print(f"Tempo de processamento do Dijkstra: {tempo_processamento:.6f} segundos\n")
    return []

def dijkstra_all_paths(rede_social, origem, destino):
    import heapq
    start_time = time.time()
    
    queue = []
    heapq.heappush(queue, (0, origem, [origem]))
    min_cost = None
    paths = []
    
    while queue:
        (cost, node, path) = heapq.heappop(queue)
        
        if min_cost is not None and cost > min_cost:
            break
        
        if node == destino:
            if min_cost is None or cost == min_cost:
                min_cost = cost
                paths.append((cost, path))
            continue
        
        for neighbor, weight in rede_social.get(node, []):
            if neighbor not in path: 
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
    
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    if not paths:
        print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Dijkstra.")
    else:
        print(f"Caminhos ótimos encontrados pelo algoritmo de Dijkstra de {origem} para {destino}:")
        for cost, path in paths:
            print(f"Caminho: {' -> '.join(path)} | Custo total: {cost}")
    print(f"Tempo de processamento do Dijkstra: {tempo_processamento:.6f} segundos\n")
    return [path for cost, path in paths]

# funções do algoritmo de floyd-warshall
def floyd_warshall(rede_social, usuarios):
    dist = {u: {v: float('inf') for v in usuarios} for u in usuarios}
    next_vertice = {u: {v: None for v in usuarios} for u in usuarios}

    for u in usuarios:
        dist[u][u] = 0
    for u in rede_social:
        for (v, peso) in rede_social[u]:
            dist[u][v] = peso
            next_vertice[u][v] = v

    for k in usuarios:
        for i in usuarios:
            for j in usuarios:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertice[i][j] = next_vertice[i][k]

    for v in usuarios:
        if dist[v][v] < 0:
            print("A rede social contém ciclos negativos. O algoritmo de Floyd-Warshall não pode ser aplicado corretamente.")
            return None, None

    return dist, next_vertice

# função do algoritmo de bellman-ford
def bellman_ford(rede_social, origem, destino):
    start_time = time.time()
    
    dist = {usuario: float('inf') for usuario in usuarios}
    predecessor = {usuario: None for usuario in usuarios}
    dist[origem] = 0
    
    for _ in range(len(usuarios) - 1):
        for u in rede_social:
            for v, peso in rede_social[u]:
                if dist[u] + peso < dist[v]:
                    dist[v] = dist[u] + peso
                    predecessor[v] = u
    
    for u in rede_social:
        for v, peso in rede_social[u]:
            if dist[u] + peso < dist[v]:
                print("A rede social contém ciclos negativos. O algoritmo de Bellman-Ford não pode ser aplicado corretamente.")
                return [], {}
    
    caminho = []
    atual = destino
    while atual is not None:
        caminho.append(atual)
        atual = predecessor[atual]
    caminho.reverse()
    
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    if dist[destino] == float('inf'):
        print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Bellman-Ford.")
    else:
        print(f"Caminho ótimo encontrado pelo algoritmo de Bellman-Ford de {origem} para {destino}:")
        print(" -> ".join(caminho))
        print(f"Custo total: {dist[destino]}")
    print(f"Tempo de processamento do Bellman-Ford: {tempo_processamento:.6f} segundos\n")
    
    return caminho, dist


def reconstruir_caminho(origem, destino, next_vertice):
    if next_vertice[origem][destino] is None:
        return None
    caminho = [origem]
    while origem != destino:
        origem = next_vertice[origem][destino]
        caminho.append(origem)
    return caminho

def executar_floyd_warshall(rede_social, origem, destino):
    start_time = time.time()
    dist, next_vertice = floyd_warshall(rede_social, usuarios)
    end_time = time.time()
    tempo_processamento = end_time - start_time
    if dist is None:
        print("A rede social contém ciclos negativos. O algoritmo de Floyd-Warshall não pode ser aplicado corretamente.")
        print(f"Tempo de processamento do Floyd-Warshall: {tempo_processamento:.6f} segundos\n")
        return []
    caminho = reconstruir_caminho(origem, destino, next_vertice)
    if caminho is None:
        print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Floyd-Warshall.")
    else:
        print(f"Caminho ótimo encontrado pelo algoritmo de Floyd-Warshall de {origem} para {destino}:")
        print(" -> ".join(caminho))
        print(f"Custo total: {dist[origem][destino]}")
    print(f"Tempo de processamento do Floyd-Warshall: {tempo_processamento:.6f} segundos\n")
    return [caminho] if caminho else []

# def identificar_bots(rede_social, caminhos, bots_conhecidos):
#     possiveis_bots = []
#     usuarios_no_caminho = set(usuario for caminho in caminhos for usuario in caminho)
#     for usuario in usuarios_no_caminho:
#         num_arestas = len(rede_social[usuario])
#         motivos = []
#         if num_arestas > 5:
#             motivos.append("muitas conexões")
#         for (vizinho, peso) in rede_social[usuario]:
#             if vizinho in bots_conhecidos:
#                 continue 
#             if peso > 10:
#                 motivos.append("interações com peso muito alto")
#                 break
#         if motivos:
#             possiveis_bots.append((usuario, motivos))
#     return possiveis_bots

# função para identificar possíveis bots com análise histórica nos caminhos
def identificar_bots_por_historico_em_caminhos(rede_social, caminhos):
    possiveis_bots = []
    usuarios_no_caminho = set(usuario for caminho in caminhos for usuario in caminho)
    
    for usuario in usuarios_no_caminho:
        conexoes = rede_social.get(usuario, [])
        num_arestas = len(conexoes)
        pesos_interacoes = [peso for _, peso in conexoes]
        motivos = []
        
        # critério 1: número excessivo de conexões
        if num_arestas > 6:
            motivos.append("muitas conexões")
        
        # critério 2: média de peso das interações anormalmente alta
        media_peso = sum(pesos_interacoes) / num_arestas if num_arestas > 0 else 0
        if media_peso > 8:
            motivos.append("interações com média de peso alta")
        
        # critério 3: discrepância entre conexões fortes e fracas
        interacoes_fortes = sum(1 for peso in pesos_interacoes if peso > 10)
        if interacoes_fortes / num_arestas > 0.5:
            motivos.append("alta proporção de interações fortes")
        
        # critério 4: presença em sub-redes suspeitas (muitos bots conhecidos)
        vizinhos_bots = sum(1 for vizinho, _ in conexoes if "Bot" in vizinho)
        if vizinhos_bots > 2:
            motivos.append("muitos vizinhos suspeitos")
        
        if motivos:
            possiveis_bots.append((usuario, motivos))
    
    return possiveis_bots

def main():
    origem = input("Digite o usuário de origem: ").strip()
    destino = input("Digite o usuário de destino: ").strip()
    
    if origem not in usuarios or destino not in usuarios:
        print("Usuários inválidos. Por favor, insira usuários existentes na rede social.")
        return
    
    print("\n--- Busca em Profundidade (DFS) ---")
    caminhos_dfs = dfs(rede_social, origem, destino)
    
    print("--- Busca em Largura (BFS) ---")
    caminhos_bfs = bfs(rede_social, origem, destino)
    
    print("--- Algoritmo de Dijkstra ---")
    caminhos_dijkstra = dijkstra_all_paths(rede_social, origem, destino)
    
    print("--- Algoritmo de Floyd-Warshall ---")
    caminhos_fw = executar_floyd_warshall(rede_social, origem, destino)

    print("--- Algoritmo de Bellman-Ford ---")
    caminho_bf, dist_bf = bellman_ford(rede_social, origem, destino)
    if caminho_bf:
        print(f"Caminho pelo Bellman-Ford: {' -> '.join(caminho_bf)} | Custo total: {dist_bf[destino]}")
    
    todos_caminhos = caminhos_dfs + caminhos_bfs + caminhos_dijkstra + caminhos_fw + [caminho_bf] if caminho_bf else []

    print("\n--- Identificação de Possíveis Bots em Caminhos ---")
    possiveis_bots = identificar_bots_por_historico_em_caminhos(rede_social, todos_caminhos)
    if possiveis_bots:
        print("Usuários possivelmente suspeitos de serem bots nos caminhos:")
        for bot, motivos in possiveis_bots:
            print(f"- {bot} (Motivos: {', '.join(motivos)})")
    else:
        print("Nenhum usuário suspeito de comportamento de bot identificado nos caminhos.")
    
    print("\n--- Visualização da Rede Social ---")
    visualizar_grafo(rede_social, [bot[0] for bot in possiveis_bots], origem, destino)

if __name__ == "__main__":
    main()
