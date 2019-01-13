from decouple import config
import matplotlib.pyplot as plt
import numpy as np
from tinydb import TinyDB, Query
import os

os.environ["DATABASE_PATH"] = r"H:\MEGH\NITK\Third Year - B.Tech NITK\DankNotDank\Memes Dataset\db.json"
# Database
PATH = config("DATABASE_PATH")
db = TinyDB(PATH)

data = db.all()

upvotes = [record['ups'] for record in data]


hist, bins, _ = plt.hist(upvotes)
plt.clf()
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(upvotes, bins=logbins)

plt.ylabel("# of Posts")
plt.xlabel("# of Upvotes")
plt.title("Distribution of Upvotes")

plt.show()

max_upvotes = max(upvotes)

print(len(upvotes))
print(max_upvotes)
upvotes_array = np.asanyarray(upvotes,dtype=None,order=None)
upvotes_array_norm = upvotes_array/max_upvotes
print(upvotes_array_norm)

np.savetxt('Normal.txt', upvotes_array_norm, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ', encoding=None)