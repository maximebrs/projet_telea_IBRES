# -*- coding: utf-8 -*-
"""
Librairie de fonctions pour le projet TELEA - Classification Supervisée
@author: Maxime Ibres
@organization: Master 2 SIGMA
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from osgeo import gdal
from libsigma import read_and_write as rw

# =======================================
# 1. UTILITAIRES FICHIERS & METADONNÉES
# =======================================

def add_nodata_metadata(path_file, nodata_value=-9999):
    """
    Ouvre l'image avec GDAL pour forcer l'écriture du tag NoData dans les métadonnées.
    """
    ds = gdal.Open(path_file, gdal.GA_Update)
    if ds:
        for i in range(ds.RasterCount):
            ds.GetRasterBand(i+1).SetNoDataValue(nodata_value)
        ds = None

def rasterize_samples(image_ref_path, vector_path, field_name):
    
    """
    Rastérise un shapefile sur la grille d'une image de référence.
    Retourne un array numpy où chaque pixel a la valeur du champ demandé.
    """
    # Ouvrir l'image de référence pour récupérer dimensions et géoréférencement
    ds_ref = gdal.Open(image_ref_path)
    if ds_ref is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {image_ref_path}")
        
    cols = ds_ref.RasterXSize
    rows = ds_ref.RasterYSize
    geo_transform = ds_ref.GetGeoTransform()
    projection = ds_ref.GetProjection()
    
    # Créer un raster temporaire en mémoire
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    
    # Ouvrir le vecteur
    shp_ds = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if shp_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {vector_path}")
    layer = shp_ds.GetLayer()
    
    # Rastériser les polygones sur le raster vide
    # Initialisation à 0
    band = target_ds.GetRasterBand(1)
    band.Fill(0)
    
    # Ecrire la valeur du champ 'strate'
    gdal.RasterizeLayer(target_ds, [1], layer, options=[f"ATTRIBUTE={field_name}"])
    
    # Récupérer le tableau numpy
    array = band.ReadAsArray()
    
    # Nettoyage
    ds_ref = None
    target_ds = None
    shp_ds = None
    
    return array

# ==============================
# 2. ANALYSE DES ECHANTILLONS
# ==============================

def analyze_polygon_metrics(gdf, class_col='strate', labels_dict=None):
    """
    Calcule les statistiques par classe (Polygones, Pixels, Moyenne).
    """
    df = gdf.copy()
    
    # Calculs géométriques
    df['area_m2'] = df.geometry.area
    df['pixels_estimes'] = df['area_m2'] / 100
    
    # Agrégation
    stats = df.groupby(class_col).agg(
        Nb_Polygones=('geometry', 'count'),
        Nb_Pixels=('pixels_estimes', 'sum')
    )
    
    # Calcul Taille Moyenne
    stats['Taille Moyenne (px)'] = stats['Nb_Pixels'] / stats['Nb_Polygones']
    
    # Mapping des noms (1 -> Sol Nu...)
    if labels_dict:
        stats.index = stats.index.map(labels_dict)
    
    # Calcul des % (sur les données propres)
    total_poly = stats['Nb_Polygones'].sum()
    total_pix = stats['Nb_Pixels'].sum()
    
    stats['% Polygones'] = (stats['Nb_Polygones'] / total_poly) * 100
    stats['% Pixels'] = (stats['Nb_Pixels'] / total_pix) * 100
    
    cols_order = ['Nb_Polygones', '% Polygones', 'Nb_Pixels', '% Pixels', 'Taille Moyenne (px)']
    return stats[cols_order]

def save_class_distributions(serie_poly, serie_pix, colors_poly, colors_pix, path_poly, path_pix):
    """
    Génère et sauvegarde les deux graphiques individuels (Polygones et Pixels)
    en respectant exactement le formatage d'origine.
    """
    # Graphique Polygones
    plt.figure(figsize=(8, 6))
    ax_p = serie_poly.plot(kind='bar', color=colors_poly)
    ax_p.bar_label(ax_p.containers[0], padding=3)
    plt.title("Répartition du nombre de polygones par classe")
    plt.ylabel("Nombre de polygones")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.ylim(0, serie_poly.max() * 1.1)
    plt.savefig(path_poly, bbox_inches='tight')
    plt.close()
    print(f"=> Graphique sauvegardé dans {path_poly}.")

    # Graphique Pixels
    plt.figure(figsize=(8, 6))
    ax_px = serie_pix.plot(kind='bar', color=colors_pix)
    ax_px.bar_label(ax_px.containers[0], fmt='{:,.0f}', padding=3)
    plt.title("Répartition du nombre de pixels (estimé) par classe")
    plt.ylabel("Nombre de pixels")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.ylim(0, serie_pix.max() * 1.1)
    plt.savefig(path_pix, bbox_inches='tight')
    plt.close()
    print(f"=> Graphique sauvegardé dans {path_pix}.")

def show_class_distributions(serie_poly, serie_pix, colors_poly, colors_pix):
    """
    Affiche la figure combinée des 2 graphiques (poly + pixels) dans le notebook.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Graphe de gauche (Polygones)
    serie_poly.plot(kind='bar', ax=ax1, color=colors_poly, rot=0)
    ax1.set_title("A. Nombre de polygones", fontsize=14)
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.set_ylim(0, serie_poly.max() * 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.bar_label(ax1.containers[0], padding=3, fontsize=11)

    # Graphe de droite (Pixels)
    serie_pix.plot(kind='bar', ax=ax2, color=colors_pix, rot=0)
    ax2.set_title("B. Nombre de pixels (estimé)", fontsize=14)
    ax2.set_ylabel("")
    ax2.set_xlabel("")
    ax2.set_ylim(0, serie_pix.max() * 1.15)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.bar_label(ax2.containers[0], fmt='{:,.0f}', padding=3, fontsize=11)

    plt.tight_layout()
    plt.show()

# ===============================
# 3. ANALYSE TEMPORELLE (NARI)
# ===============================

def compute_nari(arr_b03, arr_b05, nodata=-9999):
    """
    Calcule le NARI (Normalized Anthocyanin Reflectance Index) selon la formule :
    ((1/B03) - (1/B05)) / ((1/B03) + (1/B05))
    
    Gère les divisions par zéro et remplace les NaN par la valeur nodata.
    """
    b03 = arr_b03.astype('float32')
    b05 = arr_b05.astype('float32')
    
    # Gestion des zéros
    b03[b03 == 0] = np.nan
    b05[b05 == 0] = np.nan
    
    # Calcul
    term_b03 = 1.0 / b03
    term_b05 = 1.0 / b05
    nari = (term_b03 - term_b05) / (term_b03 + term_b05)
    
    # Nettoyage
    nari = np.nan_to_num(nari, nan=nodata)

    print("Calcul du NARI effectué.")
    
    return nari

def extract_temporal_stats(ari_cube, mask_classes, classes_info):
    """
    Extrait les statistiques temporelles (Moyenne, Std) pour chaque classe.
    Suppose que ari_cube et mask_classes ont des dimensions compatibles (H, W).
    """
    stats = {}

    for class_id, info in classes_info.items():
        # Récupération du nom et de la couleur
        class_name = info['label']
        
        # Masque booléen
        is_class = (mask_classes == class_id)
        
        # S'il y a des pixels pour cette classe
        if np.sum(is_class) > 0:
            # Extraction des valeurs
            values = ari_cube[is_class] 
            
            # Calcul des stats sur l'axe 0 (sur l'ensemble des pixels de la classe)
            stats[class_name] = {
                'mean': np.nanmean(values, axis=0),
                'std': np.nanstd(values, axis=0),
                'color': info['color']
            }
            
    return stats

def plot_nari_series(stats_dict, dates, out_filename=None):
    """
    Génère le graphique de la série temporelle moyenne d’ARI de chaque strate 
    dans le même graphique.
    """
    plt.figure(figsize=(12, 6))

    for class_name, data in stats_dict.items():
        c = data['color']
        # Tracé Moyenne
        plt.plot(dates, data['mean'], label=class_name, color=c, 
                 linewidth=2, marker='o', markersize=4)
        
        # Tracé Écart-type (zone ombrée)
        plt.fill_between(dates, 
                         data['mean'] - data['std'], 
                         data['mean'] + data['std'], 
                         color=c, alpha=0.2)

    # Mise en forme
    plt.title("Évolution temporelle moyenne de l'indice NARI par strate (Moyenne ± Ecart-type)")
    plt.ylabel("Valeur moyenne du NARI")
    plt.legend(title="Strates")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Gestion des dates en abscisse
    plt.xticks(dates, dates.strftime('%Y-%m-%d'), rotation=45)

    plt.tight_layout()
    
    if out_filename:
        plt.savefig(out_filename, dpi=150)
        print(f"=> Graphique sauvegardé : {out_filename}")
    
    plt.show()

def visualize_series(image_path, dates_list=None, cmap='YlOrRd', ncols=2):
    """
    Affiche toutes les bandes d'une image multibande (ex: Série Temporelle ARI).
    Gère le NoData et l'affichage en grille.
    """
    # Chargement
    if not os.path.exists(image_path):
        print(f"Fichier introuvable : {image_path}")
        return
    data = rw.load_img_as_array(image_path)
    
    # Gestion des dimensions
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]
    
    rows, cols, nb_bands = data.shape
    
    # Nettoyage pour l'affichage (NoData -9999 -> NaN)
    data_visu = data.astype('float32').copy()
    data_visu[data_visu == -9999] = np.nan

    # Calcul des bornes min/max GLOBALES (sur l'ensemble des dates)
    # => vmin/vmax automatique basé sur les percentiles pour éviter qu'un pixel aberrant écrase tout
    # => Garantit que la même couleur représente la même valeur partout.
    global_vmin = np.nanpercentile(data_visu, 2)
    global_vmax = np.nanpercentile(data_visu, 98)
    
    # Calcul de la grille
    nrows = int(np.ceil(nb_bands / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), constrained_layout=True)

    # Aplatir le tableau d'axes pour itérer facilement (même s'il n'y a qu'une ligne)
    if nb_bands > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    # Boucle d'affichage
    for i in range(nb_bands):
        ax = axes_flat[i]
        # Titre (Date ou Numéro de bande)
        if dates_list is not None and len(dates_list) == nb_bands:
            title = f"Date : {str(dates_list[i]).split(' ')[0]}"
        else:
            title = f"Bande {i+1}"
        # Affichage de l'image
        im = ax.imshow(
            data_visu[:, :, i], cmap=cmap, 
            vmin=global_vmin, vmax=global_vmax, 
            interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        # Barre de couleur individuelle
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Masquer les axes vides s'il y en a
    for j in range(nb_bands, len(axes_flat)):
        axes_flat[j].axis('off')

    plt.suptitle(f"Visualisation de la série : {os.path.basename(image_path)}", fontsize=16)
    plt.show()

# ======================================================================
# 4. PRÉPARATION DATASET + VISUALISATION MODÈLE & FEATURE IMPORTANCE
# ======================================================================

def prepare_training_dataset(list_files, shp_path, field_name='strate'):
    """
    Charge une liste d'images, les empile, rasterise les échantillons
    et retourne les matrices X (Features) et Y (Labels) prêtes pour Scikit-Learn.
    """
    print("--- Préparation du Dataset d'entraînement ---")
    
    features_list = []
    ref_shape = None # Pour vérifier les dimensions (Lignes, Colonnes)

    # Chargement et Empilement
    print(f"Chargement de {len(list_files)} fichiers...")
    
    for f in list_files:
        if not os.path.exists(f):
            print(f"Fichier introuvable, ignoré -> {f}")
            continue
            
        # Chargement via rw
        img = rw.load_img_as_array(f).astype('float32')
        
        # Vérification des dimensions
        if ref_shape is None:
            ref_shape = img.shape[:2]
        elif img.shape[:2] != ref_shape:
            print(f"Erreur de dimension pour {os.path.basename(f)} : {img.shape[:2]} vs {ref_shape}")
            continue 
            
        features_list.append(img)
        print(f" -> Ajouté : {os.path.basename(f)} ({img.shape[2]} dates)")

    # Concaténation finale : colle tout sur l'axe des "variables" (axe 2)
    X_full = np.concatenate(features_list, axis=2)

    # Création de la Vérité Terrain (Y)
    # La première image sert de référence géométrique
    mask_classes = rasterize_samples(list_files[0], shp_path, field_name=field_name)

    # Extraction des Pixels d'Entraînement
    # Garde que les pixels où le masque > 0 (classe existe)
    valid_pixels = (mask_classes > 0)
    # Extraction de X (Tableau des variables)
    X = X_full[valid_pixels, :]
    # Extraction de Y (Tableau des classes)
    Y = mask_classes[valid_pixels]
    
    # Nettoyage final (NaN -> 0)
    # Remplacement des NaNs par 0 pour le Random Forest
    X = np.nan_to_num(X, nan=0.0)
    X_full = np.nan_to_num(X_full, nan=0.0)

    print("Dataset, X (Tab. variables) et Y (Tab. classes) extraits.")
    
    return X, Y, X_full

def print_X_Y_matrix_bilan(X, Y, X_full):
    """
    Affiche un bilan structuré des dimensions des données
    (Image complète vs Jeu d'entraînement).
    """
    # Calculs statistiques
    total_pixels_image = X_full.shape[0] * X_full.shape[1]
    total_pixels_train = X.shape[0]
    ratio = (total_pixels_train / total_pixels_image) * 100 if total_pixels_image > 0 else 0

    print("\n" + "="*60)
    print(f"{'BILAN DES DONNÉES D ENTRÉE':^60}")
    print("="*60)

    # L'Image
    print(f"\n1. RÉALITÉ TERRAIN (X_full) -> Pour la Carte Finale")
    print(f"   • Structure  : CUBE 3D (Lignes, Colonnes, Variables)")
    print(f"   • Dimensions : {X_full.shape}")
    print(f"   • Surface    : {total_pixels_image:,} pixels totaux".replace(',', ' '))
    print(f"   • Variables  : {X_full.shape[2]}")

    print("-" * 60)

    # Le Dataset
    print(f"2. JEU D'ENTRAÎNEMENT (X, Y) -> Pour le Modèle")
    print(f"   • Structure  : TABLEAU 2D (Échantillons, Variables)")
    print(f"   • Dimensions X : {X.shape}")
    print(f"   • Dimensions Y : {Y.shape}")
    print(f"   • Échantillons : {total_pixels_train:,} pixels annotés".replace(',', ' '))

    print("-" * 60)

    # Analyse
    print(f"3. ANALYSE DU RATIO")
    print(f"   => Le modèle apprend sur {ratio:.3f}% de la zone totale.")
    
    # Vérification de cohérence (Variables)
    nb_vars_train = X.shape[1]
    nb_vars_image = X_full.shape[2]
    
    if nb_vars_train == nb_vars_image:
        print(f"   => COHÉRENCE OK : {nb_vars_train} variables de part et d'autre.")
    else:
        print(f"   => ALERTE : X a {nb_vars_train} variables mais l'image en a {nb_vars_image} !")
    
    print("="*60 + "\n")

def compute_feature_importance(model, file_list, dates_list=None):
    """
    Génère un DataFrame classant les variables par importance.
    
    Arguments:
        model      : Le modèle entraîné (ex: best_rf) contenant .feature_importances_
        file_list  : La liste des fichiers utilisés pour l'entraînement (input_files)
        dates_list : (Optionnel) Liste des dates pour nommer les bandes temporelles.
                     Doit avoir la même longueur que le nombre de bandes d'un fichier temporel.
    
    Retourne:
        DataFrame trié avec colonnes ['Variable', 'Importance']
    """
    expanded_feature_names = []

    # Génération des noms
    for f in file_list:
        filename_full = os.path.basename(f)
        filename = os.path.splitext(filename_full)[0]
        # Ouvre juste pour compter les bandes
        ds = rw.open_image(f)
        nb_bands = ds.RasterCount
        ds = None # Fermeture immédiate
        
        for i in range(nb_bands):
            if dates_list is not None and len(dates_list) == nb_bands:
                date_suffix = str(dates_list[i]).split(' ')[0] 
                expanded_feature_names.append(f"{filename}_{date_suffix}")
            else:
                expanded_feature_names.append(f"{filename}_Date{i+1}")

    # Vérifications de cohérence
    nb_names = len(expanded_feature_names)
    nb_imp = len(model.feature_importances_)
    
    print(f"Nombre de variables nommées    : {nb_names}")
    print(f"Nombre d'importances du modèle : {nb_imp}")
    
    if nb_names != nb_imp:
        print("ATTENTION : Le nombre de noms ne correspond pas au nombre de variables du modèle !")
    
    # Création du DataFrame
    df_importance = pd.DataFrame({
        'Variable': expanded_feature_names,
        'Importance': model.feature_importances_
    })
    
    # Tri décroissant
    df_sorted = df_importance.sort_values(by='Importance', ascending=False)
    
    return df_sorted

def plot_top_features(df_importance, top_n=10):
    """
    Affiche un diagramme en barres des N variables les plus importantes.
    """
    subset = df_importance.head(top_n)
    plt.figure(figsize=(10, 5))
    # [::-1] inverse l'ordre pour que la barre la plus grande soit en haut du graph
    plt.barh(subset['Variable'][::-1], 
             subset['Importance'][::-1], 
             color='skyblue')
    plt.xlabel("Importance relative")
    plt.title(f"Top {top_n} des variables les plus importantes")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_aggregated_importance(df_input):
    """
    Agrège et affiche l'importance des variables par Bande et par Date.
    Gère les noms de variables type 'bretagne_24-25_B08_2025-04-10'.
    """
    # Copie et Gestion de l'Index
    df = df_input.copy()
    
    # Si le nom de la variable est dans l'index (cas fréquent), on le sort
    if 'Feature' not in df.columns:
        df = df.reset_index()
    
    # Identification automatique des colonnes (Nom vs Valeur)
    col_name = None
    col_val = None
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_val = col
        else:
            col_name = col
            
    if col_name is None or col_val is None:
        print(f"Erreur structure : Colonnes trouvées : {df.columns}")
        return

    # Extraction
    def extract_band_date(txt):
        txt = str(txt)
        parts = txt.split('_')
        date = parts[-1]
        # Détection de la bande
        if "ARI" in txt:
            band = "ARI"
        else:
            band = "Autre"
            for p in parts:
                if p.startswith('B') and len(p) <= 3 and any(c.isdigit() for c in p):
                    band = p
                    break
        return pd.Series([band, date])

    df[['Bande', 'Date']] = df[col_name].apply(extract_band_date)
    
    # Agrégation
    band_stats = df.groupby('Bande')[col_val].sum().sort_values(ascending=True)
    date_stats = df.groupby('Date')[col_val].sum().sort_values(ascending=True)
    
    # Affichage Graphique
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique Bandes
    colors_band = plt.cm.viridis(np.linspace(0, 1, len(band_stats)))
    axes[0].barh(band_stats.index, band_stats.values, color=colors_band)
    axes[0].set_title('Importance cumulée par Bande Spectrale')
    axes[0].set_xlabel('Somme des importances')
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)

    # Graphique Dates
    colors_date = plt.cm.viridis(np.linspace(0, 1, len(date_stats)))
    axes[1].barh(date_stats.index, date_stats.values, color=colors_date)
    axes[1].set_title('Importance cumulée par Date')
    axes[1].set_xlabel('Somme des importances')
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    
    # Print console pour analyse
    print("=== TOP 3 BANDES ===")
    print(band_stats.sort_values(ascending=False).head(3))
    print("\n=== TOP 3 DATES ===")
    print(date_stats.sort_values(ascending=False).head(3))

# ===============================================
# 5. GÉNÉRATION & VISUALISATION CARTE FINALE
# ===============================================

def generate_final_map(model, X_full, ref_image_path, out_path, nodata=0):
    """
    Génère la carte classifiée finale en utilisant l'image déjà chargée (X_full).
    Ne prédit que sur les pixels valides (pas sur les bordures NoData).
    
    Arguments:
        model          : Le modèle entraîné (best_rf)
        X_full         : Le cube 3D (Lignes, Cols, Bandes) issu de prepare_training_dataset
        ref_image_path : Chemin d'une des images d'origine (pour copier la projection/GPS)
        out_path       : Chemin de sortie du fichier .tif
    """
    print(f"Génération de la carte finale...")
    print(f"Dimensions de l'image source : {X_full.shape}")
    
    rows, cols, bands = X_full.shape
    
    # Création du Masque de validité
    # Utilisation de l'image de référence sur le disque pour retrouver les vrais NaNs/NoData
    ds_ref = rw.open_image(ref_image_path)
    ref_band = ds_ref.GetRasterBand(1).ReadAsArray()
    file_nodata = ds_ref.GetRasterBand(1).GetNoDataValue()
    
    if file_nodata is not None:
        valid_mask = (ref_band != file_nodata)
    else:
        valid_mask = ~np.isnan(ref_band)
        
    print(f"Dimensions : {rows}x{cols}")
    
    # Préparation des données
    # Aplatissement (Reshape) : Transforme le cube en tableau 2D
    X_flat = X_full.reshape(rows * cols, bands)
    mask_flat = valid_mask.reshape(rows * cols)
    
    # Ne garde QUE les pixels intéressants pour le modèle
    X_to_predict = X_flat[mask_flat]
    
    # Sécurité : Remplace les NaN par 0
    X_to_predict = np.nan_to_num(X_to_predict, nan=0.0)
    
    print(f"Pixels à prédire : {X_to_predict.shape[0]} (sur {rows*cols} totaux)")
    
    # Prédiction (Uniquement sur les pixels valides)
    if X_to_predict.shape[0] > 0:
        Y_predicted = model.predict(X_to_predict)
    else:
        print("Attention : Aucun pixel valide trouvé !")
        return

    # Reconstruction de l'image complète
    # Création image vide remplie de 0
    Y_full_flat = np.zeros(rows * cols, dtype=np.uint8)
    
    # Insertion des prédictions SEULEMENT aux bons endroits
    Y_full_flat[mask_flat] = Y_predicted
    
    # Redonne la forme 2D
    carte_strates = Y_full_flat.reshape(rows, cols)
    
    # Sauvegarde
    rw.write_image(
        out_filename=out_path,
        array=carte_strates,
        data_set=ds_ref,
        gdal_dtype=gdal.GDT_Byte,
        driver_name="GTiff"
    )
    
    # Écriture propre du NoData dans les métadonnées
    add_nodata_metadata(out_path, nodata_value=nodata)
        
    print(f"Carte générée proprement : {out_path}")

# VISUALISATION DE LA CARTE FINALE

def show_classification_map(map_path, classes_info, title):
    """
    Affiche la carte finale avec les bonnes couleurs et la légende.
    
    Arguments:
        map_path     : Chemin vers le fichier .tif généré.
        classes_info : Dictionnaire contenant les labels et couleurs.
    """
    # Chargement de l'image
    ds = gdal.Open(map_path)
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    
    # Préparation des couleurs
    # Cherche l'ID max pour dimensionner la liste de couleurs
    max_id = max(classes_info.keys())
    
    # Initialise tout en blanc (pour le fond 0 et les IDs inutilisés)
    colors_list = [(1, 1, 1, 0) for _ in range(max_id + 1)]
    
    # On remplit avec nos vraies couleurs
    legend_patches = []
    
    for cls_id, info in classes_info.items():
        c = info['color']
        label = info['label']
        colors_list[cls_id] = c 
        # Prépare le carré pour la légende
        patch = mpatches.Patch(color=c, label=f"{label} (Code {cls_id})")
        legend_patches.append(patch)
        
    # Création de la Colormap personnalisée
    cmap = ListedColormap(colors_list)
    
    # Affichage
    plt.figure(figsize=(15, 10))
    plt.imshow(data, cmap=cmap, interpolation='nearest')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.axis('off') # Retire les axes X/Y
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()