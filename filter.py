import nltk    
from nltk import tokenize   
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#instalar o nltk por fora e também o pacote 'punkt'

#tokeniza sem filtrar
tokens = word_tokenize("Uma teoria que surgiu há alguns anos e vem acumulando cada vez mais evidências a seu favor questiona a velha crença de que Júpiter, o maior planeta do Sistema Solar, no lugar de servir como “escudo” protetor – ajudando a desviar corpos celestes da nossa órbita com sua colossal força gravitacional – pode, na verdade, atrair e “lançar” objetos como cometas e asteroides diretamente em rota de colisão com a Terra.")
nltk.download('stopwords')
nltk.download('punkt')
print(tokens)

#tokeniza filtrando um pouco
stop_words = set(stopwords.words('portuguese'))
tokens = [w for w in tokens if not w in stop_words]
print(tokens)
    
#tokeniza e reduz palavras e inflexões
porter = PorterStemmer()
stems = []
for t in tokens:    
    stems.append(porter.stem(t))
print(stems)

#depois precisa do TF-IDF pra verificar as palavras mais importantes
#depois Word2Vec pra contextualizar a palavras importantes do TF-IDF
