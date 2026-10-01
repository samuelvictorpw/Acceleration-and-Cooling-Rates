import plotly.express as px
import pandas as pd
import numpy as np

# 1. Lendo os dados
try:
    df = pd.read_csv('data.txt', sep=',')
    df.columns = ['Source_Name', 'Black_Hole_Mass', 'Redshift', 'L1.4', 'log_Lline']
    df['Source_Name'] = df['Source_Name'].str.strip()
except FileNotFoundError:
    print("Arquivo data.txt não encontrado! Gerando dados simulados...")
    df = pd.DataFrame({
        'Source_Name': [f'Gal_{i}' for i in range(100)], 
        'Redshift': np.random.uniform(0.1, 2, 100), 
        'Black_Hole_Mass': np.random.uniform(10, 40, 100),
        'L1.4': np.random.uniform(38, 40, 100),
        'log_Lline': np.random.uniform(41, 43, 100)
    })

# 2. Generating random RA and Dec
n_galaxies = len(df)
np.random.seed(42)
df['RA'] = np.random.uniform(0, 360, n_galaxies)
df['Dec'] = np.random.uniform(-90, 90, n_galaxies)

# 3. Creating the 3D plot
fig = px.scatter_3d(
    df,
    x='RA',             
    y='Dec',            
    z='Black_Hole_Mass',       
    size='Black_Hole_Mass',    
    size_max=40,               
    color='Redshift',          
    hover_name='Source_Name',
    custom_data=['Source_Name'], 
    hover_data={
        'Redshift': True, 
        'Black_Hole_Mass': ':.2E',
        'L1.4': True, 
        'log_Lline': True
    },
    color_continuous_scale='Inferno', 
    opacity=0.9,
    labels={
        'RA': 'Right Ascension (degree)',
        'Dec': 'Declination (degree)',
        'Redshift': 'Redshift (z)',
        'Black_Hole_Mass': 'Black Hole Mass (M☉)', 
        'L1.4': 'Luminosity 1.4 GHz (erg/s)',
        'log_Lline': 'Linear Luminosity (erg/s)'
    }
)

# --- CRIANDO O MENU SUSPENSO PARA TROCAR O EIXO Z ---
updatemenus = [
    dict(
        buttons=list([
            dict(
                args=[
                    {'z': [df['Black_Hole_Mass'].values, None], 'visible': [True, False]}, 
                    {'scene.zaxis.title.text': 'Black Hole Mass (M☉)'}
                ],
                label="Axis Z: Mass",
                method="update"
            ),
            dict(
                args=[
                    {'z': [df['Redshift'].values, None], 'visible': [True, False]}, 
                    {'scene.zaxis.title.text': 'Redshift (z)'}
                ],
                label="Axis Z: Redshift",
                method="update"
            ),
            dict(
                args=[
                    {'z': [df['L1.4'].values, None], 'visible': [True, False]}, 
                    {'scene.zaxis.title.text': 'Luminosity 1.4 GHz (erg/s)'}
                ],
                label="Axis Z: L1.4",
                method="update"
            ),
            dict(
                args=[
                    {'z': [df['log_Lline'].values, None], 'visible': [True, False]}, 
                    {'scene.zaxis.title.text': 'Linear Luminosity (erg/s)'}
                ],
                label="Axis Z: log Lline",
                method="update"
            )
        ]),
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=0.98,             
        xanchor="right",    
        y=0.98,             
        yanchor="top",
        bgcolor="rgba(30, 30, 30, 0.85)", 
        bordercolor="#555",
        font=dict(color="white")
    )
]

fig.update_layout(
    template='plotly_dark',
    scene=dict(
        xaxis=dict(backgroundcolor="black", gridcolor="#333333"),
        yaxis=dict(backgroundcolor="black", gridcolor="#333333"),
        zaxis=dict(
            backgroundcolor="black", 
            gridcolor="#333333", 
            exponentformat="power"
        )
    ),
    margin=dict(l=0, r=0, b=0, t=50),
    coloraxis_colorbar=dict(
        len=0.5,           
        yanchor="bottom",  
        y=0.05,            
        thickness=12,      
        title_side="top"
    ),
    updatemenus=updatemenus
)
fig.update_traces(marker=dict(sizemin=5))

# --- Marcador configurado como 'x' ---
fig.add_trace(
    dict(
        type='scatter3d',
        x=[df['RA'].iloc[0]],       
        y=[df['Dec'].iloc[0]], 
        z=[df['Black_Hole_Mass'].iloc[0]], 
        mode='markers+text',
        marker=dict(
            size=8,                
            color='#00ffea',        
            symbol='x',             
            line=dict(color='#00ffea', width=3) 
        ),
        text=[''],
        textfont=dict(color='#ff00ff', size=18, weight='bold'),
        textposition='top center',
        showlegend=False,
        hoverinfo='none',
        name='Target',
        visible=False               
    )
)

# 4. Saving the HTML
html_filename = "3d_galaxy_plot.html"
fig.write_html(html_filename, div_id="my_3d_plot")

# 5. GERANDO OPÇÕES PARA A BARRA DE PESQUISA
opcoes_html = "\n".join([f'        <option value="{nome}"></option>' for nome in df['Source_Name'].unique()])

# 6. JAVASCRIPT & CSS ATUALIZADOS
javascript_code = """
<style>
    /* Estilo da barra de pesquisa */
    #search-container {
        position: absolute;
        top: 20px;
        left: 20px;
        z-index: 1000;
        background-color: rgba(30, 30, 30, 0.85);
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #444;
        display: flex;
        gap: 10px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }
    #search-input {
        background-color: #111;
        color: white;
        border: 1px solid #555;
        padding: 8px 12px;
        border-radius: 4px;
        outline: none;
        width: 250px;
    }
    #search-input:focus {
        border-color: #ff9900;
    }
    #search-btn {
        background-color: #ff9900;
        color: #111;
        border: none;
        padding: 8px 15px;
        border-radius: 4px;
        font-weight: bold;
        cursor: pointer;
        transition: 0.2s;
    }
    #search-btn:hover {
        background-color: #ffaa33;
    }

    /* Estilo do menu flutuante */
    #my-dropdown-menu {
        display: none;
        position: fixed; 
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #444;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.7);
        z-index: 9999;
        font-family: Arial, sans-serif;
        min-width: 240px; 
    }
    #my-dropdown-menu h4 {
        margin: 0 0 10px 0;
        font-size: 14px;
        text-align: center;
        color: #ff9900;
        border-bottom: 1px solid #444;
        padding-bottom: 5px;
    }
    .menu-title {
        color: #00ffea;
        font-size: 13px;
        margin: 5px 0 10px 0;
        text-align: center;
    }
    .menu-btn {
        display: block;
        width: 100%;
        background-color: #333;
        color: #e0e0e0;
        border: 1px solid #555;
        padding: 8px;
        margin-bottom: 6px;
        border-radius: 4px;
        cursor: pointer;
        text-align: center;
        transition: background 0.2s;
    }
    .menu-btn:hover {
        background-color: #555;
        color: white;
    }
    
    /* Layout para botões lado a lado */
    .flex-row {
        display: flex;
        gap: 6px;
        margin-bottom: 6px;
    }
    .flex-row .menu-btn {
        margin-bottom: 0;
        flex: 1;
        font-size: 13px;
    }

    .close-btn {
        background-color: #8b0000;
        margin-top: 10px;
        border-color: #ff0000;
    }
    .close-btn:hover {
        background-color: #b22222;
    }
    .folder-btn {
        background-color: #444;
    }
    .length-btn {
        background-color: #225544;
    }
</style>

<!-- Barra de Pesquisa HTML -->
<div id="search-container">
    <input type="text" id="search-input" list="source-list" placeholder="Search Source Name...">
    <datalist id="source-list">
""" + opcoes_html + """
    </datalist>
    <button id="search-btn" onclick="searchSource()">Search</button>
</div>

<!-- Menu flutuante HTML -->
<div id="my-dropdown-menu">
    <h4>Source: <span id="source-name-menu"></span></h4>
    
    <!-- Opções Iniciais (Menu Principal) -->
    <div id="main-menu-options">
        <button class="menu-btn" onclick="selectParticle('electrons')">⚡ Electrons</button>
        <button class="menu-btn" onclick="openSubMenu('pn-menu-options')">⚛️ Protons & Neutrinos</button>
        <button class="menu-btn" onclick="openSubMenu('nuclei-menu-options')">🪐 Nuclei</button>
        <button class="menu-btn close-btn" onclick="closeMenu()">❌ Close</button>
    </div>

    <!-- Submenu: Prótons e Neutrinos -->
    <div id="pn-menu-options" style="display: none;">
        <div class="menu-title">Protons & Neutrinos</div>
        <div class="flex-row">
            <button class="menu-btn" onclick="selectParticle('protons')">⚛️ Protons</button>
            <button class="menu-btn" onclick="selectParticle('neutrinos')">👻 Neutrinos</button>
        </div>
        <button class="menu-btn close-btn" onclick="goBack()">⬅️ Back</button>
    </div>

    <!-- Submenu: Núcleos -->
    <div id="nuclei-menu-options" style="display: none;">
        <div class="menu-title">Heavy Nuclei</div>
        <div class="flex-row">
            <button class="menu-btn" onclick="selectParticle('oxygen')">🫧 Oxygen</button>
            <button class="menu-btn" onclick="selectParticle('iron')">🧲 Iron</button>
        </div>
        <!-- Botão reservado para o próximo núcleo -->
        <button class="menu-btn" onclick="selectParticle('helium')" style="border: 1px dashed #777; color: #aaa;">🎈 Helium</button>
        <button class="menu-btn close-btn" onclick="goBack()">⬅️ Back</button>
    </div>

    <!-- Opções de Pasta (Cenários de Proporção) -->
    <div id="folder-menu-options" style="display: none;">
        
        <div class="menu-title" style="color: #ff9900;">1. Spectra / Flux:</div>
        <button class="menu-btn folder-btn" onclick="openFinalImage('01', 'flux')">🔴 Electrons > Protons</button>
        <button class="menu-btn folder-btn" onclick="openFinalImage('02', 'flux')">🟠 Electrons = Protons</button>
        <button class="menu-btn folder-btn" onclick="openFinalImage('03', 'flux')">🟢 Electrons &lt; Protons</button>
        
        <!-- SEÇÃO LENGTH SCALES -->
        <div id="length-scales-section" style="display: none; margin-top: 15px;">
            <div class="menu-title" style="color: #00ffea; border-top: 1px solid #444; padding-top: 10px;">2. Length Scales:</div>
            <div class="flex-row">
                <button class="menu-btn length-btn" onclick="openFinalImage('01', 'length')">📏 01</button>
                <button class="menu-btn length-btn" onclick="openFinalImage('02', 'length')">📏 02</button>
                <button class="menu-btn length-btn" onclick="openFinalImage('03', 'length')">📏 03</button>
            </div>
        </div>

        <button class="menu-btn close-btn" onclick="goBack()">⬅️ Back</button>
    </div>
</div>

<script>
    var currentSource = "";
    var currentParticleType = ""; 
    
    // Variáveis para histórico de navegação
    var currentMenu = 'main-menu-options';
    var menuHistory = [];

    var mouseX = 0;
    var mouseY = 0;
    document.addEventListener('mousemove', function(e) {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    var mainTraceIndex = 0;
    var targetTraceIndex = 1; 

    // ---- Funções de Navegação dos Menus ----
    function showMenu(menuId) {
        document.getElementById('main-menu-options').style.display = 'none';
        document.getElementById('pn-menu-options').style.display = 'none';
        document.getElementById('nuclei-menu-options').style.display = 'none';
        document.getElementById('folder-menu-options').style.display = 'none';
        
        document.getElementById(menuId).style.display = 'block';
    }

    function openSubMenu(menuId) {
        menuHistory.push(currentMenu);
        currentMenu = menuId;
        showMenu(menuId);
    }

    function selectParticle(type) {
        currentParticleType = type;
        
        // Verifica se é núcleo para exibir os botões de Length Scales
        var lsSection = document.getElementById('length-scales-section');
        if (type === 'oxygen' || type === 'iron' || type === 'helium') {
            lsSection.style.display = 'block';
        } else {
            lsSection.style.display = 'none';
        }

        menuHistory.push(currentMenu);
        currentMenu = 'folder-menu-options';
        showMenu('folder-menu-options');
    }

    function goBack() {
        if (menuHistory.length > 0) {
            currentMenu = menuHistory.pop();
            showMenu(currentMenu);
        }
    }

    function resetMenus() {
        menuHistory = [];
        currentMenu = 'main-menu-options';
        showMenu('main-menu-options');
    }
    // ----------------------------------------

    function decodeArray(val) {
        if (Array.isArray(val)) return val;
        if (val && typeof val === 'object' && typeof val.bdata === 'string') {
            var bin = atob(val.bdata);
            var len = bin.length;
            var buf = new ArrayBuffer(len);
            var view = new Uint8Array(buf);
            for (var i = 0; i < len; i++) view[i] = bin.charCodeAt(i);
            var TypedArrayCtor;
            switch (val.dtype) {
                case 'f8': TypedArrayCtor = Float64Array; break;
                case 'f4': TypedArrayCtor = Float32Array; break;
                case 'i4': TypedArrayCtor = Int32Array; break;
                case 'i2': TypedArrayCtor = Int16Array; break;
                case 'i1': TypedArrayCtor = Int8Array; break;
                case 'u4': TypedArrayCtor = Uint32Array; break;
                case 'u2': TypedArrayCtor = Uint16Array; break;
                case 'u1': TypedArrayCtor = Uint8Array; break;
                default: TypedArrayCtor = Float64Array;
            }
            return Array.prototype.slice.call(new TypedArrayCtor(buf));
        }
        return val;
    }

    function clearHighlight() {
        var myPlot = document.getElementById('my_3d_plot');
        Plotly.restyle(myPlot, { visible: [false] }, [targetTraceIndex]);
    }

    function highlightByIndex(idx) {
        var myPlot = document.getElementById('my_3d_plot');
        var trace = myPlot.data[mainTraceIndex];

        var xs = decodeArray(trace.x);
        var ys = decodeArray(trace.y);
        var zs = decodeArray(trace.z);

        var ptX = xs[idx];
        var ptY = ys[idx];
        var ptZ = zs[idx];

        Plotly.restyle(myPlot, {
            x: [[ptX]],
            y: [[ptY]],
            z: [[ptZ]],
            text: [['🎯 ' + currentSource]],
            visible: [true] 
        }, [targetTraceIndex]);
    }

    // --- FUNÇÃO ATUALIZADA COM NOME EXATO ---
    function openFinalImage(folderNumber, plotCategory) {
        if (!currentSource || !currentParticleType) return;
        
        var path = "";
        
        // Se o usuário clicou nos botões de "Length Scales"
        if (plotCategory === 'length') {
            
            // Monta o nome do arquivo exatamente como você pediu: Fig_NOME.png
            var fileName = "Fig_" + currentSource + ".png"; 

            if (currentParticleType === "oxygen") {
                path = "oxigenio/fig7_plots/" + folderNumber + "/" + fileName;
            } else if (currentParticleType === "iron") {
                path = "Iron/fig7_plots/" + folderNumber + "/" + fileName;
            } else if (currentParticleType === "helium") {
                path = "helium/fig7_plots/" + folderNumber + "/" + fileName;
            }
            
        } 
        // Se o usuário clicou nos botões normais de Fluxo/Espectro
        else {
            if (currentParticleType === "neutrinos") {
                path = "image/" + folderNumber + "/neutrino_flux_" + currentSource + ".png";
            } else if (currentParticleType === "oxygen") {
                path = "oxigenio/image/" + folderNumber + "/" + currentSource + "_oxygen.png";
            } else if (currentParticleType === "iron") {
                path = "Iron/image/" + folderNumber + "/" + currentSource + "_iron.png";
            }else if (currentParticleType === "helium") {
                path = "helium/image/" + folderNumber + "/" + currentSource + "_helium.png";
            } else {
                path = "image/" + folderNumber + "/" + currentSource + "_" + currentParticleType + ".png";
            }
        }

        window.open(path, '_blank');
        closeMenu(); 
    }

    function closeMenu() {
        document.getElementById('my-dropdown-menu').style.display = 'none';
        document.getElementById('search-input').value = ''; 
        clearHighlight(); 
        resetMenus(); 
    }

    function searchSource() {
        var inputVal = document.getElementById('search-input').value.trim();
        if (!inputVal) return;
        
        clearHighlight(); 
        resetMenus(); 

        var myPlot = document.getElementById('my_3d_plot');
        var found = false;
        var foundIdx = -1;

        var trace = myPlot.data[mainTraceIndex];
        if (trace.customdata) {
            for (var i = 0; i < trace.customdata.length; i++) {
                if (trace.customdata[i][0] === inputVal) {
                    found = true;
                    foundIdx = i;
                    break;
                }
            }
        }

        if (found) {
            currentSource = inputVal;
            document.getElementById('source-name-menu').textContent = currentSource;

            highlightByIndex(foundIdx);

            var menu = document.getElementById('my-dropdown-menu');
            menu.style.transform = 'none';
            menu.style.right = 'auto'; 
            menu.style.left = '20px';
            menu.style.top = '80px'; 
            menu.style.display = 'block';
        } else {
            alert("Source '" + inputVal + "' not found! Please check the name.");
        }
    }

    document.getElementById('search-input').addEventListener('keypress', function (e) {
        if (e.key === 'Enter') searchSource();
    });

    var checkPlotly = setInterval(function() {
        var myPlot = document.getElementById('my_3d_plot');
        
        if (myPlot && typeof myPlot.on === 'function') {
            clearInterval(checkPlotly); 
            
            var menu = document.getElementById('my-dropdown-menu');
            var nameSpan = document.getElementById('source-name-menu');

            myPlot.on('plotly_click', function(data){
                if(data.points && data.points.length > 0 && data.points[0].curveNumber === mainTraceIndex) {
                    var point = data.points[0];
                    currentSource = point.customdata[0];
                    nameSpan.textContent = currentSource;
                    
                    resetMenus(); 

                    menu.style.transform = 'none';
                    menu.style.right = 'auto'; 
                    menu.style.left = (mouseX + 15) + 'px';
                    menu.style.top = (mouseY + 15) + 'px';
                    menu.style.display = 'block';
                }
            });
        }
    }, 200);
</script>
"""

# Usando "a" (append) para adicionar os botões na página sem apagar o plot
with open(html_filename, "a", encoding="utf-8") as file:
    file.write(javascript_code)

print("Plot atualizado com Sucesso! 🚀")