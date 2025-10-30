import numpy as np
import random
import time

def greedy_path(cost_matrix, start=None):
    N = cost_matrix.shape[0]
    if start is None:
        start = random.randrange(N)
    visited = [start]
    unvisited = set(range(N))
    unvisited.remove(start)
    cur = start
    while unvisited:
        next_j = min(unvisited, key=lambda j: cost_matrix[cur, j])
        visited.append(next_j)
        unvisited.remove(next_j)
        cur = next_j
    return visited

def path_cost(path, cost_matrix):
    total = 0.0
    for i in range(len(path)-1):
        total += cost_matrix[path[i], path[i+1]]
    return total

def two_opt(path, cost_matrix, max_iter=1000):
    N = len(path)
    improved = True
    it = 0
    best = path[:]
    best_cost = path_cost(best, cost_matrix)
    while improved and it < max_iter:
        improved = False
        it += 1
        for i in range(1, N-2):
            for j in range(i+1, N):
                if j - i == 1:  # adjacent, skip
                    continue
                new_path = best[:i] + best[i:j][::-1] + best[j:]
                new_cost = path_cost(new_path, cost_matrix)
                if new_cost < best_cost:
                    best = new_path
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return best, best_cost

def multi_start_greedy(cost_matrix, starts=10):
    N = cost_matrix.shape[0]
    best_path = None
    best_cost = float('inf')
    for s in range(min(starts, N)):
        path = greedy_path(cost_matrix, start=s)
        p, c = two_opt(path, cost_matrix, max_iter=200)
        if c < best_cost:
            best_cost = c
            best_path = p
    return best_path, best_cost
