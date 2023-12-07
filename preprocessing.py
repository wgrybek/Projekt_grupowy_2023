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

cat = df[["key","audio_mode", "time_signature"]].copy()
num = df.drop(["song_name","key","audio_mode", "time_signature"], axis = 1)
Y = df["song_popularity"]

# # Wizualizacja

#Numerical variables
fig = plt.figure(figsize = (15,15))
ax = fig.gca()
num.hist(ax = ax)
plt.show()

# Categorical values
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,5))
cat["key"].value_counts().plot(kind='bar', ax = ax1)
cat["time_signature"].value_counts().plot(kind='bar', ax = ax2)
cat["audio_mode"].value_counts().plot(kind='bar', ax = ax3)
plt.show()

# # Zmienne kategoryczne

Counter(df['time_signature'])

# usuwamy wiersze gdzie wartość zmiennej time_signature wynosi 0 i 1
df.drop(df[ df['time_signature'] == 1 ].index, inplace = True)
df.drop(df[ df['time_signature'] == 0 ].index, inplace = True)
Counter(df['time_signature'])

Counter(df['audio_mode'])

Counter(df['key'])

# # Korelacje

# +
correlation_matrix=num.corr()

plt.figure(figsize=(12,10))
ax = sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r')
ax.xaxis.tick_top()
plt.xticks(rotation=90)
plt.show()
# -

# # Zależności między zmienną objaśnianą a zmiennymi objaśniającymi

# +
#numeric
fig, axes = plt.subplots(5, 2, figsize = (15,20))
plt.subplots_adjust(hspace=0.5)
num.plot.scatter(x = "song_popularity",
                 y = "song_duration_ms",
                 ax = axes[0,0])
num.plot.scatter(x = "song_popularity",
                 y = "acousticness",
                 ax = axes[0,1])
num.plot.scatter(x = "song_popularity",
                 y = "danceability",
                 ax = axes[1,0])
num.plot.scatter(x = "song_popularity",
                 y = "energy",
                 ax = axes[1,1])
num.plot.scatter(x = "song_popularity",
                 y = "instrumentalness",
                 ax = axes[2,0])
num.plot.scatter(x = "song_popularity",
                 y = "liveness",
                 ax = axes[2,1])
num.plot.scatter(x = "song_popularity",
                 y = "loudness",
                 ax = axes[3,0])
num.plot.scatter(x = "song_popularity",
                 y = "speechiness",
                 ax = axes[3,1])
num.plot.scatter(x = "song_popularity",
                 y = "tempo",
                 ax = axes[4,0])
num.plot.scatter(x = "song_popularity",
                 y = "audio_valence",
                 ax = axes[4,1])

plt.show()

# -

#categorical
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (15,5))
sns.boxplot(x="key", y="song_popularity", data=df, palette="Set3", ax = ax1)
sns.boxplot(x="audio_mode", y="song_popularity", data=df, palette="Set3", ax = ax2)
sns.boxplot(x="time_signature", y="song_popularity", data=df, palette="Set3", ax = ax3)
plt.show()


# # Normalizacja (numeric)

scaler = StandardScaler()
X = num.drop(["song_popularity"], axis = 1)
X.iloc[:,0:]=scaler.fit_transform(X.iloc[:,0:].to_numpy())

# # Pary wykresy (numeric)

sns.pairplot(X)

# # PCA (numeric)

pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X)
print('Procent warincji wyjaśniony przez components: {}'.format(pca.explained_variance_ratio_))
print('Procent warincji wyjaśniony przez components (suma): {}'.format(pca.explained_variance_ratio_.cumsum()))


