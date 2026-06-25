"""
Модуль визуализации для ESP алгоритма.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch


def create_network_visualization(genome_list, n_inputs, n_hidden, n_outputs, 
                                  filename='network_structure.png', 
                                  title='RNN Structure'):
    """
    Визуализирует структуру рекуррентной нейронной сети.
    
    Args:
        genome_list: список геномов (по одному на каждый скрытый нейрон)
        n_inputs: количество входов
        n_hidden: количество скрытых нейронов
        n_outputs: количество выходов
        filename: имя файла для сохранения
        title: заголовок графика
    """
    
    # Создаем ориентированный граф
    G = nx.DiGraph()
    
    # Добавляем узлы
    # Входные нейроны (0..n_inputs-1)
    for i in range(n_inputs):
        G.add_node(f'input_{i}', layer='input', index=i)
    
    # Скрытые нейроны (0..n_hidden-1)
    for i in range(n_hidden):
        G.add_node(f'hidden_{i}', layer='hidden', index=i)
    
    # Выходные нейроны (0..n_outputs-1)
    for i in range(n_outputs):
        G.add_node(f'output_{i}', layer='output', index=i)
    
    # Добавляем ребра и веса
    edges = []
    edge_weights = []
    edge_colors = []
    
    genome_size = n_inputs + n_hidden + 1 + n_outputs
    
    for i in range(n_hidden):
        g = genome_list[i]
        idx = 0
        
        # Веса от входов к скрытому нейрону i
        w_in = g[idx : idx + n_inputs]; idx += n_inputs
        for j in range(n_inputs):
            G.add_edge(f'input_{j}', f'hidden_{i}', weight=w_in[j])
            edges.append((f'input_{j}', f'hidden_{i}'))
            edge_weights.append(abs(w_in[j]))
            edge_colors.append('red' if w_in[j] > 0 else 'blue')
        
        # Рекуррентные веса от скрытых нейронов к скрытому нейрону i
        w_hid = g[idx : idx + n_hidden]; idx += n_hidden
        for j in range(n_hidden):
            if i != j:  # Не рисуем петли (хотя можно и их)
                G.add_edge(f'hidden_{j}', f'hidden_{i}', weight=w_hid[j])
                edges.append((f'hidden_{j}', f'hidden_{i}'))
                edge_weights.append(abs(w_hid[j]))
                edge_colors.append('red' if w_hid[j] > 0 else 'blue')
        
        # Bias (пропускаем визуализацию bias для простоты)
        idx += 1
        
        # Веса от скрытого нейрона i к выходам
        w_out = g[idx : idx + n_outputs]
        for j in range(n_outputs):
            G.add_edge(f'hidden_{i}', f'output_{j}', weight=w_out[j])
            edges.append((f'hidden_{i}', f'output_{j}'))
            edge_weights.append(abs(w_out[j]))
            edge_colors.append('red' if w_out[j] > 0 else 'blue')
    
    # Позиции узлов
    pos = {}
    
    # Входные нейроны слева
    for i in range(n_inputs):
        pos[f'input_{i}'] = (0, 1 - i/(n_inputs-1) if n_inputs > 1 else 0.5)
    
    # Скрытые нейроны в центре
    for i in range(n_hidden):
        pos[f'hidden_{i}'] = (1, 1 - i/(n_hidden-1) if n_hidden > 1 else 0.5)
    
    # Выходные нейроны справа
    for i in range(n_outputs):
        pos[f'output_{i}'] = (2, 1 - i/(n_outputs-1) if n_outputs > 1 else 0.5)
    
    # Рисуем граф
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Рисуем узлы по слоям
    input_nodes = [n for n in G.nodes() if G.nodes[n]['layer'] == 'input']
    hidden_nodes = [n for n in G.nodes() if G.nodes[n]['layer'] == 'hidden']
    output_nodes = [n for n in G.nodes() if G.nodes[n]['layer'] == 'output']
    
    nx.draw_networkx_nodes(G, pos, nodelist=input_nodes, node_color='lightblue', 
                          node_size=1500, label='Input', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=hidden_nodes, node_color='lightgreen', 
                          node_size=1500, label='Hidden (RNN)', ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=output_nodes, node_color='lightcoral', 
                          node_size=1500, label='Output', ax=ax)
    
    # Рисуем подписи узлов
    labels = {}
    for node in G.nodes():
        if node.startswith('input'):
            labels[node] = f'In {node.split("_")[1]}'
        elif node.startswith('hidden'):
            labels[node] = f'H {node.split("_")[1]}'
        elif node.startswith('output'):
            labels[node] = f'Out {node.split("_")[1]}'
    
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)
    
    # Рисуем ребра с разной толщиной в зависимости от веса
    nx.draw_networkx_edges(G, pos, edgelist=edges, 
                          width=[max(0.5, w*2) for w in edge_weights],
                          edge_color=edge_colors, 
                          arrows=True, arrowsize=20, 
                          arrowstyle='->', 
                          ax=ax,
                          alpha=0.7)
    
    # Добавляем легенду
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Positive weight'),
        Line2D([0], [0], color='blue', lw=2, label='Negative weight'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
               markersize=15, label='Input layer'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
               markersize=15, label='Hidden layer (RNN)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
               markersize=15, label='Output layer')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Визуализация сети сохранена в {filename}")


def create_genome_statistics(genome_list, n_inputs, n_hidden, n_outputs, 
                             filename='genome_statistics.png'):
    """
    Визуализирует статистику геномов.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    genome_size = n_inputs + n_hidden + 1 + n_outputs
    
    # 1. Распределение весов
    all_weights = []
    for g in genome_list:
        all_weights.extend(g)
    
    axes[0, 0].hist(all_weights, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_weights), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(all_weights):.3f}')
    axes[0, 0].axvline(np.std(all_weights), color='green', linestyle='--', 
                      label=f'Std: {np.std(all_weights):.3f}')
    axes[0, 0].set_title('Distribution of All Weights')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Средняя абсолютная величина весов для каждого скрытого нейрона
    neuron_avg_weights = []
    for i, g in enumerate(genome_list):
        avg_weight = np.mean(np.abs(g))
        neuron_avg_weights.append(avg_weight)
    
    axes[0, 1].bar(range(n_hidden), neuron_avg_weights, color='lightgreen', 
                   edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Average Absolute Weight per Hidden Neuron')
    axes[0, 1].set_xlabel('Hidden Neuron Index')
    axes[0, 1].set_ylabel('Average |Weight|')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Тепловая карта весов (входы -> скрытые)
    input_hidden_weights = np.zeros((n_hidden, n_inputs))
    for i in range(n_hidden):
        g = genome_list[i]
        input_hidden_weights[i, :] = g[:n_inputs]
    
    im = axes[1, 0].imshow(input_hidden_weights, cmap='RdBu_r', aspect='auto')
    axes[1, 0].set_title('Weights: Input → Hidden')
    axes[1, 0].set_xlabel('Input Index')
    axes[1, 0].set_ylabel('Hidden Neuron Index')
    plt.colorbar(im, ax=axes[1, 0], label='Weight Value')
    
    # 4. Тепловая карта рекуррентных весов (скрытые -> скрытые)
    recurrent_weights = np.zeros((n_hidden, n_hidden))
    for i in range(n_hidden):
        g = genome_list[i]
        recurrent_weights[i, :] = g[n_inputs:n_inputs+n_hidden]
    
    im2 = axes[1, 1].imshow(recurrent_weights, cmap='RdBu_r', aspect='auto')
    axes[1, 1].set_title('Recurrent Weights: Hidden → Hidden')
    axes[1, 1].set_xlabel('From Hidden Neuron')
    axes[1, 1].set_ylabel('To Hidden Neuron')
    plt.colorbar(im2, ax=axes[1, 1], label='Weight Value')
    
    plt.suptitle(f'Genome Statistics (Genome Size = {genome_size})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Статистика геномов сохранена в {filename}")