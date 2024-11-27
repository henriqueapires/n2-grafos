import time
import matplotlib.pyplot as plt
import networkx as nx

# modelagem do grafo representando a rede social
# cada nó representa um usuário, e as arestas representam interações entre usuários, com pesos indicando a frequência das interações

rede_social = {
    'A': [('B', 4), ('C', 1), ('D', 7), ('Bot1', 8)],  # bot1 conectado diretamente a a
    'B': [('A', 4), ('E', 3), ('F', 6), ('Bot2', 10)],  # bot2 conectado diretamente a b
    'C': [('A', 1), ('D', 2), ('G', 5)],  
    'D': [('A', 7), ('C', 2), ('E', 3), ('H', 4), ('Bot3', 9)],  # bot3 conectado a d
    'E': [('B', 3), ('D', 3), ('F', 2), ('I', 6)],  
    'F': [('B', 6), ('E', 2), ('J', 3), ('Bot4', 11)],  # bot4 conectado a f
    'G': [('C', 5), ('H', 2), ('J', 4)],  
    'H': [('D', 4), ('G', 2), ('I', 1)],  
    'I': [('E', 6), ('H', 1), ('J', 5)],  
    'J': [('F', 3), ('G', 4), ('I', 5), ('Bot5', 13)],  # bot5 conectado a j
    'Bot1': [('A', 8), ('Bot6', 12)],  # bot1 conectado à rede principal e outro bot
    'Bot2': [('B', 10), ('Bot7', 14)],  # bot2 conectado à rede principal e outro bot
    'Bot3': [('D', 9), ('Bot8', 15)],  # bot3 conectado à rede principal e outro bot
    'Bot4': [('F', 11), ('Bot6', 10)],  # bot4 conectado à rede principal e outro bot
    'Bot5': [('J', 13), ('Bot7', 12)],  # bot5 conectado à rede principal e outro bot
    'Bot6': [('Bot1', 12), ('Bot4', 10)],  # bot6 conectado entre bots
    'Bot7': [('Bot2', 14), ('Bot5', 12)],  # bot7 conectado entre bots
    'Bot8': [('Bot3', 15)],  # bot8 apenas conectado a bot3
}

usuarios = list(rede_social.keys())

# função para visualizar a lista de adjacências como um grafo
def visualizar_grafo(rede_social, possiveis_bots, origem, destino):
    # cria um grafo direcionado vazio usando a biblioteca networkx
    G = nx.DiGraph()
    
    # adiciona as arestas e nós ao grafo com base nas conexões da rede social
    for usuario, conexoes in rede_social.items():
        for vizinho, peso in conexoes:
            G.add_edge(usuario, vizinho, weight=peso)
    
    # define a posição dos nós no gráfico de forma que a visualização fique agradável
    pos = nx.spring_layout(G)
    # define o tamanho da figura para a plotagem
    plt.figure(figsize=(12, 10))
    
    # cria uma lista para armazenar as cores dos nós
    node_colors = []
    # percorre todos os nós do grafo para definir suas cores
    for node in G.nodes():
        if node == origem:
            node_colors.append("green")       # origem em verde
        elif node == destino:
            node_colors.append("orange")      # destino em laranja
        elif node in possiveis_bots:
            node_colors.append("red")         # possíveis bots em vermelho
        else:
            node_colors.append("skyblue")     # demais nós em azul claro
    
    # desenha o grafo com os parâmetros definidos
    nx.draw_networkx(G, pos, with_labels=True, node_size=400, node_color=node_colors, font_size=8, font_weight="bold", arrows=True)
    # obtém os pesos das arestas para exibir nos labels
    labels = nx.get_edge_attributes(G, 'weight')
    # desenha os labels das arestas com os pesos
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # define o título do gráfico
    plt.title("Rede Social - Representação Visual do Grafo")
    # exibe o gráfico
    plt.show()

# função de busca em profundidade (dfs)
def dfs(rede_social, origem, destino):
    # inicia a contagem do tempo de processamento
    start_time = time.time()
    # lista para armazenar os caminhos encontrados
    caminhos = []
    
    # função recursiva interna para realizar a busca em profundidade
    def dfs_recursive(usuario_atual, caminho):
        # verifica se o usuário atual é o destino
        if usuario_atual == destino:
            # se for, adiciona o caminho encontrado à lista de caminhos
            caminhos.append(caminho)
            return
        # percorre os vizinhos do usuário atual
        for (vizinho, _) in rede_social.get(usuario_atual, []):
            # se o vizinho não estiver no caminho atual (para evitar ciclos)
            if vizinho not in caminho:
                # chama recursivamente a função para o vizinho
                dfs_recursive(vizinho, caminho + [vizinho])
                    
    # inicia a busca em profundidade a partir da origem
    dfs_recursive(origem, [origem])
    # calcula o tempo total de processamento
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    # verifica se algum caminho foi encontrado
    if not caminhos:
        print(f"não existe caminho de {origem} para {destino} usando dfs.")
    else:
        print(f"caminhos encontrados usando dfs de {origem} para {destino}:")
        for caminho in caminhos:
            print(" -> ".join(caminho))
        # encontra o caminho ótimo (menor em termos de número de arestas)
        caminho_otimo = min(caminhos, key=len)
        print(f"caminho ótimo: {' -> '.join(caminho_otimo)}")
    print(f"tempo de processamento do dfs: {tempo_processamento:.6f} segundos\n")
    return caminhos

# função de busca em largura (bfs)
def bfs(rede_social, origem, destino):
    # inicia a contagem do tempo de processamento
    start_time = time.time()
    # lista para armazenar os caminhos encontrados
    caminhos = []
    from collections import deque
    
    # cria uma fila para armazenar os nós a serem explorados
    fila = deque()
    # adiciona a origem na fila com o caminho inicial
    fila.append((origem, [origem]))
    
    # enquanto houver nós na fila
    while fila:
        # retira o primeiro elemento da fila
        (usuario, caminho) = fila.popleft()
        # percorre os vizinhos do usuário atual
        for (vizinho, _) in rede_social.get(usuario, []):
            # se o vizinho não estiver no caminho atual (para evitar ciclos)
            if vizinho not in caminho:
                # cria um novo caminho incluindo o vizinho
                novo_caminho = caminho + [vizinho]
                # se o vizinho for o destino, adiciona o caminho à lista de caminhos
                if vizinho == destino:
                    caminhos.append(novo_caminho)
                else:
                    # caso contrário, adiciona o vizinho na fila para continuar a busca
                    fila.append((vizinho, novo_caminho))
                        
    # calcula o tempo total de processamento
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    # verifica se algum caminho foi encontrado
    if not caminhos:
        print(f"não existe caminho de {origem} para {destino} usando bfs.")
    else:
        print(f"caminhos encontrados usando bfs de {origem} para {destino}:")
        for caminho in caminhos:
            print(" -> ".join(caminho))
        # encontra o caminho ótimo (menor em termos de número de arestas)
        caminho_otimo = min(caminhos, key=len)
        print(f"caminho ótimo: {' -> '.join(caminho_otimo)}")
    print(f"tempo de processamento do bfs: {tempo_processamento:.6f} segundos\n")
    return caminhos

# função do algoritmo de dijkstra
def dijkstra(rede_social, origem, destino):
    # inicia a contagem do tempo de processamento
    start_time = time.time()
    # verifica se existem pesos negativos no grafo
    for arestas in rede_social.values():
        for (_, peso) in arestas:
            if peso < 0:
                print("a rede social contém interações com pesos negativos. o algoritmo de dijkstra não pode ser aplicado.")
                return []
    
    import heapq
    
    # cria uma fila de prioridades (heap) para armazenar os nós a serem explorados
    fila = []
    # adiciona a origem na fila com custo zero
    heapq.heappush(fila, (0, origem, [origem]))
    # conjunto para armazenar os nós já visitados
    visitados = set()
    
    # enquanto houver nós na fila
    while fila:
        # retira o nó com o menor custo acumulado
        (custo, usuario, caminho) = heapq.heappop(fila)
        # se o nó já foi visitado, ignora
        if usuario in visitados:
            continue
        # marca o nó como visitado
        visitados.add(usuario)
    
        # se o usuário atual é o destino, termina a busca
        if usuario == destino:
            end_time = time.time()
            tempo_processamento = end_time - start_time
            print(f"caminho ótimo encontrado pelo algoritmo de dijkstra de {origem} para {destino}:")
            print(" -> ".join(caminho))
            print(f"custo total: {custo}")
            print(f"tempo de processamento do dijkstra: {tempo_processamento:.6f} segundos\n")
            return [caminho]
    
        # percorre os vizinhos do usuário atual
        for (vizinho, peso) in rede_social.get(usuario, []):
            # se o vizinho ainda não foi visitado
            if vizinho not in visitados:
                # adiciona o vizinho na fila com o custo atualizado
                heapq.heappush(fila, (custo + peso, vizinho, caminho + [vizinho]))
    
    # se não encontrou caminho
    end_time = time.time()
    tempo_processamento = end_time - start_time
    print(f"não existe caminho de {origem} para {destino} usando o algoritmo de dijkstra.")
    print(f"tempo de processamento do dijkstra: {tempo_processamento:.6f} segundos\n")
    return []

def dijkstra_all_paths(rede_social, origem, destino):
    import heapq
    # inicia a contagem do tempo de processamento
    start_time = time.time()
    
    # inicialização da fila de prioridades e outras variáveis
    queue = []
    heapq.heappush(queue, (0, origem, [origem]))
    min_cost = None
    paths = []
    
    # enquanto houver nós na fila
    while queue:
        # retira o nó com o menor custo acumulado
        (cost, node, path) = heapq.heappop(queue)
        
        # se já encontramos um caminho com custo menor, podemos parar
        if min_cost is not None and cost > min_cost:
            break
        
        # se o nó atual é o destino
        if node == destino:
            # verifica se o custo é igual ao menor custo encontrado
            if min_cost is None or cost == min_cost:
                min_cost = cost
                paths.append((cost, path))
            continue
        
        # percorre os vizinhos do nó atual
        for neighbor, weight in rede_social.get(node, []):
            # se o vizinho não está no caminho atual (para evitar ciclos)
            if neighbor not in path:
                # adiciona o vizinho na fila com o custo atualizado
                heapq.heappush(queue, (cost + weight, neighbor, path + [neighbor]))
        
    # calcula o tempo total de processamento
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    # verifica se encontrou caminhos
    if not paths:
        print(f"não existe caminho de {origem} para {destino} usando o algoritmo de dijkstra.")
    else:
        print(f"caminhos ótimos encontrados pelo algoritmo de dijkstra de {origem} para {destino}:")
        for cost, path in paths:
            print(f"caminho: {' -> '.join(path)} | custo total: {cost}")
    print(f"tempo de processamento do dijkstra: {tempo_processamento:.6f} segundos\n")
    return [path for cost, path in paths]

# funções do algoritmo de floyd-warshall
def floyd_warshall(rede_social, usuarios):
    # inicializa as matrizes de distância e próximo vértice
    dist = {u: {v: float('inf') for v in usuarios} for u in usuarios}
    next_vertice = {u: {v: None for v in usuarios} for u in usuarios}
    
    # define a distância de cada nó para si mesmo como zero
    for u in usuarios:
        dist[u][u] = 0
    # inicializa as distâncias e próximos vértices com base nas arestas existentes
    for u in rede_social:
        for (v, peso) in rede_social[u]:
            dist[u][v] = peso
            next_vertice[u][v] = v
    
    # algoritmo principal do floyd-warshall
    for k in usuarios:
        for i in usuarios:
            for j in usuarios:
                # se a distância através de k é menor, atualiza
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertice[i][j] = next_vertice[i][k]
    
    # verifica se há ciclos negativos
    for v in usuarios:
        if dist[v][v] < 0:
            print("a rede social contém ciclos negativos. o algoritmo de floyd-warshall não pode ser aplicado corretamente.")
            return None, None
    
    return dist, next_vertice

# função do algoritmo de bellman-ford
def bellman_ford(rede_social, origem, destino):
    # inicia a contagem do tempo de processamento
    start_time = time.time()
    
    # inicialização das distâncias e predecessores
    dist = {usuario: float('inf') for usuario in usuarios}
    predecessor = {usuario: None for usuario in usuarios}
    dist[origem] = 0
    
    # relaxamento das arestas
    for _ in range(len(usuarios) - 1):  # realiza |V| - 1 iterações
        for u in rede_social:
            for v, peso in rede_social[u]:
                if dist[u] + peso < dist[v]:
                    dist[v] = dist[u] + peso
                    predecessor[v] = u
        
    # verificação de ciclos negativos
    for u in rede_social:
        for v, peso in rede_social[u]:
            if dist[u] + peso < dist[v]:
                print("a rede social contém ciclos negativos. o algoritmo de bellman-ford não pode ser aplicado corretamente.")
                return [], {}
        
    # reconstrução do caminho a partir dos predecessores
    caminho = []
    atual = destino
    while atual is not None:
        caminho.append(atual)
        atual = predecessor[atual]
    caminho.reverse()
    
    # calcula o tempo total de processamento
    end_time = time.time()
    tempo_processamento = end_time - start_time
    
    # verifica se existe caminho até o destino
    if dist[destino] == float('inf'):
        print(f"não existe caminho de {origem} para {destino} usando o algoritmo de bellman-ford.")
    else:
        print(f"caminho ótimo encontrado pelo algoritmo de bellman-ford de {origem} para {destino}:")
        print(" -> ".join(caminho))
        print(f"custo total: {dist[destino]}")
    print(f"tempo de processamento do bellman-ford: {tempo_processamento:.6f} segundos\n")
    
    return caminho, dist


def reconstruir_caminho(origem, destino, next_vertice):
    # se não houver próximo vértice entre origem e destino, não há caminho
    if next_vertice[origem][destino] is None:
        return None
    caminho = [origem]
    # reconstrói o caminho usando a matriz next_vertice
    while origem != destino:
        origem = next_vertice[origem][destino]
        caminho.append(origem)
    return caminho

def executar_floyd_warshall(rede_social, origem, destino):
    # inicia a contagem do tempo de processamento
    start_time = time.time()
    dist, next_vertice = floyd_warshall(rede_social, usuarios)
    # calcula o tempo total de processamento
    end_time = time.time()
    tempo_processamento = end_time - start_time
    if dist is None:
        print("a rede social contém ciclos negativos. o algoritmo de floyd-warshall não pode ser aplicado corretamente.")
        print(f"tempo de processamento do floyd-warshall: {tempo_processamento:.6f} segundos\n")
        return []
    # reconstrói o caminho ótimo entre origem e destino
    caminho = reconstruir_caminho(origem, destino, next_vertice)
    if caminho is None:
        print(f"não existe caminho de {origem} para {destino} usando o algoritmo de floyd-warshall.")
    else:
        print(f"caminho ótimo encontrado pelo algoritmo de floyd-warshall de {origem} para {destino}:")
        print(" -> ".join(caminho))
        print(f"custo total: {dist[origem][destino]}")
    print(f"tempo de processamento do floyd-warshall: {tempo_processamento:.6f} segundos\n")
    return [caminho] if caminho else []

# função para identificar possíveis bots com análise histórica nos caminhos
def identificar_bots_por_historico_em_caminhos(rede_social, caminhos):
    # lista para armazenar possíveis bots identificados
    possiveis_bots = []
    # obtém o conjunto de usuários presentes nos caminhos
    usuarios_no_caminho = set(usuario for caminho in caminhos for usuario in caminho)
    
    for usuario in usuarios_no_caminho:
        # obtém as conexões do usuário
        conexoes = rede_social.get(usuario, [])
        num_arestas = len(conexoes)
        # lista dos pesos das interações
        pesos_interacoes = [peso for _, peso in conexoes]
        motivos = []
        
        # critério 1: número excessivo de conexões
        if num_arestas > 6:
            motivos.append("muitas conexões")
        
        # critério 2: média de peso das interações anormalmente alta
        media_peso = sum(pesos_interacoes) / num_arestas if num_arestas > 0 else 0
        if media_peso > 8:
            motivos.append("interações com média de peso alta")
        
        # critério 3: alta proporção de interações com peso elevado
        interacoes_fortes = sum(1 for peso in pesos_interacoes if peso > 10)
        if interacoes_fortes / num_arestas > 0.5:  # mais da metade são interações fortes
            motivos.append("alta proporção de interações fortes")
        
        # critério 4: muitos vizinhos suspeitos (bots conhecidos)
        vizinhos_bots = sum(1 for vizinho, _ in conexoes if "Bot" in vizinho)
        if vizinhos_bots > 2:
            motivos.append("muitos vizinhos suspeitos")
        
        if motivos:
            possiveis_bots.append((usuario, motivos))
    
    return possiveis_bots

# substituição da chamada de identificar_bots no main
def main():
    # solicita ao usuário para digitar a origem e o destino
    origem = input("Digite o usuário de origem: ").strip()
    destino = input("Digite o usuário de destino: ").strip()
    
    # verifica se os usuários existem na rede social
    if origem not in usuarios or destino not in usuarios:
        print("usuários inválidos. por favor, insira usuários existentes na rede social.")
        return
    
    print("\n--- Busca em Profundidade (DFS) ---")
    # executa o algoritmo de busca em profundidade
    caminhos_dfs = dfs(rede_social, origem, destino)
    
    print("--- Busca em Largura (BFS) ---")
    # executa o algoritmo de busca em largura
    caminhos_bfs = bfs(rede_social, origem, destino)
    
    print("--- Algoritmo de Dijkstra ---")
    # executa o algoritmo de dijkstra para encontrar todos os caminhos ótimos
    caminhos_dijkstra = dijkstra_all_paths(rede_social, origem, destino)
    
    print("--- Algoritmo de Floyd-Warshall ---")
    # executa o algoritmo de floyd-warshall
    caminhos_fw = executar_floyd_warshall(rede_social, origem, destino)

    print("--- Algoritmo de Bellman-Ford ---")
    # executa o algoritmo de bellman-ford
    caminho_bf, dist_bf = bellman_ford(rede_social, origem, destino)
    if caminho_bf:
        print(f"caminho pelo bellman-ford: {' -> '.join(caminho_bf)} | custo total: {dist_bf[destino]}")
    
    # combina todos os caminhos encontrados
    todos_caminhos = caminhos_dfs + caminhos_bfs + caminhos_dijkstra + caminhos_fw + [caminho_bf] if caminho_bf else []

    print("\n--- Identificação de Possíveis Bots em Caminhos ---")
    # identifica possíveis bots nos caminhos usando análise histórica
    possiveis_bots = identificar_bots_por_historico_em_caminhos(rede_social, todos_caminhos)
    if possiveis_bots:
        print("usuários possivelmente suspeitos de serem bots nos caminhos:")
        for bot, motivos in possiveis_bots:
            print(f"- {bot} (motivos: {', '.join(motivos)})")
    else:
        print("nenhum usuário suspeito de comportamento de bot identificado nos caminhos.")
    
    print("\n--- Visualização da Rede Social ---")
    # visualiza o grafo da rede social destacando possíveis bots, origem e destino
    visualizar_grafo(rede_social, [bot[0] for bot in possiveis_bots], origem, destino)

if __name__ == "__main__":
    main()
