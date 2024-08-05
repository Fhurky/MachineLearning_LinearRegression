import pandas as pd
import matplotlib.pyplot as plt
import time
import random

# Veri seti
veri_seti = "study_score_decreasing.csv" # study_score_decreasing.csv # study_score_increasing.csv 
data = pd.read_csv(veri_seti)

# Gradient Descent Fonksiyonu
def gradient_descent(m_next, b_next, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].study_time
        y = points.iloc[i].score
        
        m_gradient += -(2/n) * x * (y - (m_next * x + b_next))
        b_gradient += -(2/n) * (y - (m_next * x + b_next))
    
    m = m_next - m_gradient * 0.0001 #(L = 0.0001)
    b = b_next - b_gradient * 0.1    #(L = 0.1)
    
    return m, b

# Grafik Gösterim Fonksiyonu
def show_graph(m, b):
    plt.scatter(data.study_time, data.score, color="red")
    x_range = range(int(data.study_time.min()), int(data.study_time.max()) + 1)
    plt.plot(x_range, [m * x + b for x in x_range], color="blue")
    plt.xlabel('Study Time')
    plt.ylabel('Score')
    plt.title('Study Time vs Score')
    plt.show()
    time.sleep(0.001)
    print("=>  F(X):", round(m, 1), "X +", round(b, 3))

# Ana Fonksiyon
def main(m, b, L, epochs):
    print("=>  F(X):", m, "X", b)
    
    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)
        show_graph(m, b)
        
# Başlangıç değerleri
main(random.uniform(-1, 110), random.uniform(-10, 10), 0.1, 250)


