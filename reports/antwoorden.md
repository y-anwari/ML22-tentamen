# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?
- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?

Antwoord:
- Deze architectuur is interessant voor experimentele doeleinden, maar niet voor commerciële doeleinden, omdat het hierbij om een simpele structuur gaat. Zo kan er gewerkt worden aan de features, waaruit veel kansen te bieden heeft om de performance van het model te verbeteren. 
- Voordelen: training verloopt sneller, makkelijk op te zetten. Nadelen: kans op overfitting, resultaten niet realistisch voor commerciële doeleinden

- Over algemeen ziet de input en output van het model prima uit. Het loggen van de resultaten in een apart document zorgt ervoor dat we mogelijke fouten kunnen achterhalen. Ik ben niet tevreden over de aantal nodes binnen hidden layers, bij h1 staat het op 100 en h2 naar 10 om weer naar 20 te gaan als output. H2 had hoger moeten liggen dan de output, maar lager dan de input. Als H2 op 40 stond dan is er een kans dat het model beter generaliseert ten opzichte van 10. De dropoutrate staat op 50%, dit houdt in dat elke keer 50% van de units weggehaald worden als er getraind wordt. Dropout inregelen is handig, hierdoor ontstaat er geen specialisme binnen het model. Echter, 50% is veel te hoog, elke keer wordt de helft van de te trainen units weggegooid, hierdoor kan het model minder goed worden getraind. Vuistregel is tussen 0-4. In dit geval zou ik eerder voor 0.2 of 0.3 gaan. 

## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)
- Hoe had hij dit ook kunnen oplossen?
- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

Antwoord:
- Het is een alternatief voor data pooling. Het reduceert de dimensies van de activation maps. Als dit niet gebruikt wordt, kan dit leiden tot overfitting. 
- Door de klasse MaxPool1D te gebruiken van torch.nn. 
- Voordelen: het werkt ook bij meerdere dimensies. Nadelen: het is lastig te achterhalen waarom deze fout kan gaan. 


### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.
- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.
- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).

Antwoord:
De architecturen die ik zou willen overwegen zijn LSTM en GRU met een attention laag. Hidden Size tussen 64 en 128, hoger zou ik niet kiezen. De dataset is zeer gering en relatief eenvoudig te classifiseren. LSTM heeft mijn voorkeur als de dataset groter zou zijn. Dat komt omdat deze getraind is op meer features dan GRU. Hierdoor zijn er meer paramaters beschikbaar, waardoor het model betere predicties kan uitvoeren. 

Bij dit architectuur is er rekening gehouden met de hoeveelheid data. Naarmate de dataset toeneemt, wordt er een uitgebreidere architectuur verwacht. Dat is natuurlijk ook afhankelijk hoeveel tijd je hebt om te experimenteren en hoe snel je computer jouw model traint. Omdat het hierbij om een tijdreekserie gaat, zou RNN zich bij uitstek goed lenen om gebruikt te worden. In dit geval vormt GRU de meest voor de handliggende keuze. Dit model werkt snel en levert in de beste resultaten op als het gaat om kleine datasets. LSTM was ook een optie geweest, mits de datasets groot waren. Dat is nu niet het geval. GRU is dus minder complex dan LSTM, omdat dit model minder poorten heeft, maar levert tegelijkertijd de beste resultaten op. De architectuur die ik hiervoor zou willen gaan gebruiken is, 64 hidden size, aantal layers zou ik op 3 houden en de dropout zou ik op 0.4 houden. Zoals hierboven benoemd, de dataset en het opgegeven probeel is niet erg complex en met de onderstaande instellingen, zou het model moeten werken. Een hogere hidden size en of layers maakt het model onnodig complex, wat meer tijd kost om te trainen en het mogelijkerwijs ook gaat overfitten. 

### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

Antwoord:

- GRUmodel, hidden Size 64, aantal layers 3 en dropout 0.4 scoort: 95% op accuracy 
- GRUmodel, hidden Size 64, aantal layers 3 en dropout 0.1 scoort: 95% op accuracy 
- GRUmodel, hidden Size 128, aantal layers 3 en dropout 0.4 scoort: 96% op accuracy 
- GRUmodel, hidden Size 128, aantal layers 3 en dropout 0.1 scoort: 94% op accuracy

Het GRU model met een attention laag en hidden size 128, aantal layers 3 en dropoutrate van 40% levert de beste score. De accuracy van het model bedraagt (afgerond) 96%, dit i.t.t. lineair model van maar liefst 73% accuracy. Dit model scoort dus beduidend beter. De overige parameters scoren min of meer hetzelfde. Dit is een goede basis om het model te kunnen hypertunen. Dit is ook goed terug te zien in figuur 1 en 2. De viertal modellen hebben geen overfitting. 

<figure>
  <p align = "center">
    <img src="/home/azureuser/code/ML22-tentamen/reports/img/Scherm­afbeelding 2023-02-14 om 22.18.25.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 1. De bestscorende configuraties</b>
    </figcaption>
  </p>
</figure>

<figure>
  <p align = "center">
    <img src="/home/azureuser/code/ML22-tentamen/reports/img/Scherm­afbeelding 2023-02-14 om 22.18.44.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 2. De meest accurate modellen</b>
    </figcaption>
  </p>
</figure>

## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je data iets in de dataloader aanpast aanpassen, doe het dan ook en laat zien wat je gedaan hebt. 
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.


Antwoord:
Op basis van het antwoord op vraag 1d kan er geconcludeerd worden dat het aantal layers tussen 64 en 128 moet zijn om te kunnen hypertunen. Verder valt het op dat de dropoutrat in ieder geval niet hoger moet zijn dan 40%. Dit moet dus tussen 10%-40% liggen. Het aantal layers heb ik voor het gemak op 3 gehouden, omdat het een relatief eenvoudig probleem is en dit aantal layer afdoende is.


### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

Antwoord:
In figuur 3 is te zien dat het model met de hoogste accuracy tussen 90% en 95%. Opvallen is dat het aantal layers in de eerste instantie 3 was maar door hypertunen 4 layers toch beter uitkomt. Bij een classificatiemodel over een audiofile zijn drie lagen afdoende, was de verwachting. Het aantal hiden size ligt tussen 120-125. En de dropoutrat bevindt zich tussen 6% en 11%. 

<figure>
  <p align = "center">
    <img src="/home/azureuser/code/ML22-tentamen/reports/img/Scherm­afbeelding 2023-02-14 om 22.43.37.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 3. Het gehypertunede prijswinnede model</b>
    </figcaption>
  </p>
</figure>

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

Op basis van het bovenstaand komt de prijswinnende settings uit op:

dropout%: 0.11028
hidden size: 126
aantal layers: 4
epochs: 130
accuracy: 95%

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 

