import time

grafo = {
    'A': [('B', 4), ('C', 1), ('D', 7)],
    'B': [('A', 4), ('E', 3), ('F', 6)],
    'C': [('A', 1), ('D', 2), ('G', 5), ('Bot1', 12)],  
    'D': [('A', 7), ('C', 2), ('E', 3), ('H', 4)],
    'E': [('B', 3), ('D', 3), ('F', 2), ('I', 6)],
    'F': [('B', 6), ('E', 2), ('J', 3)],
    'G': [('C', 5), ('H', 2), ('J', 4)],
    'H': [('D', 4), ('G', 2), ('I', 1), ('Bot2', 15)],  
    'I': [('E', 6), ('H', 1), ('J', 5)],
    'J': [('F', 3), ('G', 4), ('I', 5), ('Bot1', 10)],  
    'Bot1': [('C', 12), ('J', 10)],  
    'Bot2': [('H', 15)],              
}
vertices = list(grafo.keys())

def dfs(grafo, origem, destino):
    start_time = time.time()
    caminhos = []
    pilha = [(origem, [origem])]

    while pilha:
        (vertice, caminho) = pilha.pop()
        for (vizinho, _) in grafo.get(vertice, []):
            if vizinho not in caminho:
                if vizinho == destino:
                    caminhos.append(caminho + [vizinho])
                else:
                    pilha.append((vizinho, caminho + [vizinho]))

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

def bfs(grafo, origem, destino):
    start_time = time.time()
    from collections import deque

    caminhos = []
    fila = deque()
    fila.append((origem, [origem]))

    while fila:
        (vertice, caminho) = fila.popleft()
        for (vizinho, _) in grafo.get(vertice, []):
            if vizinho not in caminho:
                if vizinho == destino:
                    caminhos.append(caminho + [vizinho])
                else:
                    fila.append((vizinho, caminho + [vizinho]))

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

def dijkstra(grafo, origem, destino):
    start_time = time.time()
    for arestas in grafo.values():
        for (_, peso) in arestas:
            if peso < 0:
                print("O grafo contém pesos negativos. O algoritmo de Dijkstra não pode ser aplicado.")
                return

    import heapq

    fila = []
    heapq.heappush(fila, (0, origem, [origem]))
    visitados = set()

    while fila:
        (custo, vertice, caminho) = heapq.heappop(fila)
        if vertice in visitados:
            continue
        visitados.add(vertice)

        if vertice == destino:
            end_time = time.time()
            tempo_processamento = end_time - start_time
            print(f"Caminho ótimo encontrado pelo algoritmo de Dijkstra de {origem} para {destino}:")
            print(" -> ".join(caminho))
            print(f"Custo total: {custo}")
            print(f"Tempo de processamento do Dijkstra: {tempo_processamento:.6f} segundos\n")
            return

        for (vizinho, peso) in grafo.get(vertice, []):
            if vizinho not in visitados:
                heapq.heappush(fila, (custo + peso, vizinho, caminho + [vizinho]))

    end_time = time.time()
    tempo_processamento = end_time - start_time
    print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Dijkstra.")
    print(f"Tempo de processamento do Dijkstra: {tempo_processamento:.6f} segundos\n")

def floyd_warshall(grafo, vertices):
    dist = {u: {v: float('inf') for v in vertices} for u in vertices}
    next_vertice = {u: {v: None for v in vertices} for u in vertices}

    for u in vertices:
        dist[u][u] = 0
    for u in grafo:
        for (v, peso) in grafo[u]:
            dist[u][v] = peso
            next_vertice[u][v] = v

    for k in vertices:
        for i in vertices:
            for j in vertices:
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertice[i][j] = next_vertice[i][k]

    for v in vertices:
        if dist[v][v] < 0:
            print("O grafo contém ciclos negativos. O algoritmo de Floyd-Warshall não pode ser aplicado corretamente.")
            return None, None

    return dist, next_vertice

def reconstruir_caminho(origem, destino, next_vertice):
    if next_vertice[origem][destino] is None:
        return None
    caminho = [origem]
    while origem != destino:
        origem = next_vertice[origem][destino]
        caminho.append(origem)
    return caminho

def executar_floyd_warshall(grafo, origem, destino):
    start_time = time.time()
    dist, next_vertice = floyd_warshall(grafo, vertices)
    end_time = time.time()
    tempo_processamento = end_time - start_time
    if dist is None:
        print(f"O grafo contém ciclos negativos. O algoritmo de Floyd-Warshall não pode ser aplicado corretamente.")
        print(f"Tempo de processamento do Floyd-Warshall: {tempo_processamento:.6f} segundos\n")
        return
    caminho = reconstruir_caminho(origem, destino, next_vertice)
    if caminho is None:
        print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Floyd-Warshall.")
    else:
        print(f"Caminho ótimo encontrado pelo algoritmo de Floyd-Warshall de {origem} para {destino}:")
        print(" -> ".join(caminho))
        print(f"Custo total: {dist[origem][destino]}")
    print(f"Tempo de processamento do Floyd-Warshall: {tempo_processamento:.6f} segundos\n")

def bellman_ford(grafo, vertices, origem, destino):
    start_time = time.time()
    dist = {v: float('inf') for v in vertices}
    anterior = {v: None for v in vertices}
    dist[origem] = 0

    for _ in range(len(vertices) - 1):
        for u in grafo:
            for (v, peso) in grafo[u]:
                if dist[u] + peso < dist[v]:
                    dist[v] = dist[u] + peso
                    anterior[v] = u

    for u in grafo:
        for (v, peso) in grafo[u]:
            if dist[u] + peso < dist[v]:
                print("O grafo contém ciclos negativos. O algoritmo de Bellman-Ford não pode ser aplicado.")
                return

    caminho = []
    atual = destino
    while atual is not None:
        caminho.insert(0, atual)
        atual = anterior[atual]

    end_time = time.time()
    tempo_processamento = end_time - start_time

    if dist[destino] == float('inf'):
        print(f"Não existe caminho de {origem} para {destino} usando o algoritmo de Bellman-Ford.")
    else:
        print(f"Caminho ótimo encontrado pelo algoritmo de Bellman-Ford de {origem} para {destino}:")
        print(" -> ".join(caminho))
        print(f"Custo total: {dist[destino]}")
    print(f"Tempo de processamento do Bellman-Ford: {tempo_processamento:.6f} segundos\n")

def main():
    origem = input("Digite o vértice de origem: ").strip()
    destino = input("Digite o vértice de destino: ").strip()

    if origem not in vertices or destino not in vertices:
        print("Vértices inválidos. Por favor, insira vértices existentes no grafo.")
        return

    print("\n--- Busca em Profundidade (DFS) ---")
    dfs(grafo, origem, destino)

    print("--- Busca em Largura (BFS) ---")
    bfs(grafo, origem, destino)

    print("--- Algoritmo de Dijkstra ---")
    dijkstra(grafo, origem, destino)

    print("--- Algoritmo de Floyd-Warshall ---")
    executar_floyd_warshall(grafo, origem, destino)

    print("--- Algoritmo de Bellman-Ford ---")
    bellman_ford(grafo, vertices, origem, destino)

if __name__ == "__main__":
    main()
