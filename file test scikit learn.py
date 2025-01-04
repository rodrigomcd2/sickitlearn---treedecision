import pandas as pd # importa a biblioteca
from sklearn.model_selection import train_test_split # divide os dados e treina eles
from sklearn.tree import DecisionTreeClassifier, plot_tree # treina o modelo e decide q é uma arvore de decisão
from sklearn.metrics import classification_report, accuracy_score # calcula a acurancia do modelo e gera o modelo
from sklearn.preprocessing import LabelEncoder # transforma os dados em numeros
import matplotlib.pyplot as plt # visualização grafica

# DataFrame com dados de exemplo
df_treino = pd.DataFrame({
    'faixa_etaria': [20, 25, 27, 30, 35, 40, 50, 55, 60],#idade do cliente
    'poder_compra': [1, 2, 3, 3, 4, 5, 1, 2, 5],# 1 é o menor valor e 5 é o maior valor
    'genero': ['M', 'M', 'M', 'F', 'F', 'F', 'M', 'F', 'F'],# masculino e feminino
    'tipo_produto': ['A', 'B', 'B', 'C', 'C', 'C', 'A', 'B', 'C']
})

df_treino['genero'] = df_treino['genero'].map({'M': 0, 'F': 1}) # altera os valores em em 0 e 1

# Codificando a variável-alvo 'tipo_produto'
label_encoder = LabelEncoder()
df_treino['tipo_produto_encoded'] = label_encoder.fit_transform(df_treino['tipo_produto'])


X = df_treino[['faixa_etaria', 'poder_compra', 'genero']] # Dividindo as variáveis em (X) e a variável alvo (y)
y = df_treino['tipo_produto_encoded']

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criando o modelo da Árvore de Decisão
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Prevendo os resultados
y_pred = model.predict(X_test)

# Avaliando o modelo e gerando o relatorio
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Visualizando a imagem da  arvore de Decisão
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=['faixa_etaria', 'poder_compra', 'genero'],
          class_names=label_encoder.classes_, filled=True, rounded=True)
plt.show()
