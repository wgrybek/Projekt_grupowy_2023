# # Biblioteki

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# # Get data

csv_file = "song_data.csv"
df = pd.read_csv(csv_file)


# # Opis danych
#
# **song_name** : *object* - nazwa piosenki
#
# **song_popularity** : *integer* - Popularność utworu. Wartość będzie mieścić się w przedziale od 0 do 100, przy czym 100 oznacza największą popularność. Popularność jest obliczana za pomocą algorytmu i opiera się głównie na całkowitej liczbie odtworzeń utworu i na tym, jak niedawno te odtworzenia miały miejsce. Ogólnie rzecz biorąc, utwory, które są obecnie często odtwarzane, będą miały wyższą popularność niż utwory, które były często odtwarzane w przeszłości. Zduplikowane utwory (np. ten sam utwór z singla i albumu) są oceniane niezależnie. Popularność artysty i albumu jest wyznaczana matematycznie na podstawie popularności utworu.
#
# **song_duration_ms** : *integer* - Czas trwania piosenki w milisekundach.
#
# **acousticness** : *float* - Mówi o tym jak dużą mamy pewność, że piosenka jest akustyczna. 1.0 oznacza dużą pewność.
#
# **danceability** : *float* - Ocena jak bardzo taneczna jest piosenka, na podstawie kombinacji aspektów muzycznych, takich jak tempo, rytm, beat. 0.0 to najmniejsza taneczność a 1.0 to największa.
#
# **energy** : *float* - Mierzy energiczność piosenki w oparciu o dynamikę, głośność, barwę dżwięku, częstotliwość i ogólną entropię.
#
# **instrumentalness** : *float* - Przewidywanie czy muzyka nie zawiera wokalu. Im bliżej do 1.0 tym większe prawdopodobieństwo, że piosenka nie zawiera wokalu. Wartości powyżej 0.5 mają reprezentować utwory intrumentalne.
#
# **key** : *integer* - Tonacja utworu. 0 to C, 1 to C#/D♭, itd. Jeżeli tonacja jest nie podana przyjmuje wartość -1.
#
# **liveness** : *float* - Wykrywa obecność publiczności w nagraniu. Wyższe wartości żywotności reprezentują zwiększone prawdopodobieństwo, że utwór został wykonany na żywo. Wartość powyżej 0,8 stanowi silne prawdopodobieństwo, że utwór jest nagraniem na żywo.
#
# **audio_mode** : *binary* - Tryb wskazuje modalność (durową lub molową) utworu, czyli rodzaj skali, z której pochodzi jego treść melodyczna. Durowa jest reprezentowana przez 1, a molowa przez 0.
#
# **speechiness** : *float* - Wykrywa obecność mówionych słów w utworze. Im bardziej nagranie przypomina mowę (np. talk-show, audiobook, poezja), tym bliżej wartość atrybutu jest do 1,0. Wartości powyżej 0,66 opisują utwory, które prawdopodobnie składają się całkowicie z mówionych słów. Wartości między 0,33 a 0,66 opisują utwory, które mogą zawierać zarówno muzykę, jak i mowę, zarówno w sekcjach, jak i warstwach, w tym przypadku muzyki rap. Wartości poniżej 0,33 najprawdopodobniej reprezentują muzykę i inne utwory niespokojne.
#
# **tempo** : *float* - Ogólnie szacowane tempo utworu wyrażone w uderzeniach na minutę (BPM). Przykład: 118.211 BPM
#
# **time_signature** : *integer* - Szacowane metrum utworu. Metrum określa, ile uderzeń zawiera każda takt (czyli każda "miara"). Metrum jest wyrażane jako liczba uderzeń w takt, na przykład "3/4" lub "7/4".
#
# **audio_valence** : *float* - Miara od 0.0 do 1.0 opisująca muzyczną pozytywność przekazywaną przez utwór. Utwory z wysoka wartością brzmią bardziej pozytywnie (np. szczęśliwie, radośnie, euforycznie), podczas gdy utwory o niskiej mierze brzmią bardziej negatywnie (np. smutnie, przygnębiono, gniewnie).
#
# W przypadku zmiennych **key** oraz **tempo**, konieczne będzie użycie techniki One Hot Encoding, w celu zapisu ich jako zmienne binarne.

df.info()

# # Usuwanie duplikatów

counter = Counter(df['song_name'])
if len(Counter({k: c for k, c in counter.items() if c > 1})) != 0:
  df = df.drop_duplicates(subset='song_name')
  df = df.reset_index(drop=True)

# # Statystyki

df.describe().transpose()

# # Wizualizacja

fig = plt.figure(figsize = (15,15))
ax = fig.gca()
df.hist(ax = ax)
plt.show()



# # Zmienne kategoryczne

Counter(df['time_signature'])

# usuwamy wiersze gdzie wartość zmiennej time_signature wynosi 0 i 1
df.drop(df[ df['time_signature'] == 1 ].index, inplace = True)
df.drop(df[ df['time_signature'] == 0 ].index, inplace = True)
Counter(df['time_signature'])

Counter(df['audio_mode'])

Counter(df['key'])

labels, counts = np.unique(df['audio_mode'],return_counts=True)
ticks = range(len(counts))
plt.bar(ticks,counts, align='center', )
plt.xticks(ticks, labels)

labels, counts = np.unique(df['key'],return_counts=True)
ticks = range(len(counts))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, labels)

labels, counts = np.unique(df['time_signature'],return_counts=True)
ticks = range(len(counts))
plt.bar(ticks,counts, align='center')
plt.xticks(ticks, labels)

# # Korelacje

# +
X = df.drop(labels=['song_popularity', 'song_name'], axis=1)
correlation_matrix=X.corr()

plt.figure(figsize=(12,10))
ax = sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r')
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.show()
# -

# # Normalizacja

scaler = StandardScaler()
X.iloc[:,0:]=scaler.fit_transform(X.iloc[:,0:].to_numpy())

sns.pairplot(X)

# # PCA

pca = PCA(n_components=13)
principalComponents = pca.fit_transform(X)
print('Procent warincji wyjaśniony przez components: {}'.format(pca.explained_variance_ratio_))
print('Procent warincji wyjaśniony przez components (suma): {}'.format(pca.explained_variance_ratio_.cumsum()))


