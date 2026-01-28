# -*- coding: utf-8 -*-
"""
Librairie de fonctions pour le projet TELEA - Classification Supervisée
@author: Maxime Ibres
@organization: Master SIGMA
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
    C'est une bonne pratique pour que QGIS/ArcGIS reconnaisse le fond transparent.
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
    # 1. Ouvrir l'image de référence pour récupérer dimensions et géoréférencement
    ds_ref = gdal.Open(image_ref_path)
    if ds_ref is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {image_ref_path}")
        
    cols = ds_ref.RasterXSize
    rows = ds_ref.RasterYSize
    geo_transform = ds_ref.GetGeoTransform()
    projection = ds_ref.GetProjection()
    
    # 2. Créer un raster temporaire en mémoire
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    
    # 3. Ouvrir le vecteur
    shp_ds = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if shp_ds is None:
        raise FileNotFoundError(f"Impossible d'ouvrir {vector_path}")
    layer = shp_ds.GetLayer()
    
    # 4. Rastériser les polygones sur le raster vide
    # Initialisation à 0
    band = target_ds.GetRasterBand(1)
    band.Fill(0)
    
    # Ecrire la valeur du champ 'strate'
    gdal.RasterizeLayer(target_ds, [1], layer, options=[f"ATTRIBUTE={field_name}"])
    
    # 5. Récupérer le tableau numpy
    array = band.ReadAsArray()
    
    # Nettoyage
    ds_ref = None
    target_ds = None
    shp_ds = None
    
    return array

# ==============================
# 2. ANALYSE DES ECHANTILLONS
# ==============================

def save_class_distributions(serie_poly, serie_pix, colors_poly, colors_pix, path_poly, path_pix):
    """
    Génère et sauvegarde les deux graphiques individuels (Polygones et Pixels)
    en respectant exactement le formatage d'origine.
    """
    # A. Graphique Polygones (Sauvegarde)
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

    # B. Graphique Pixels (Sauvegarde)
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
            # Si ari_cube est (H, W, D), values sera (N_pixels, D)
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
    Génère le graphique de la série temporelle moyenne d’ARI de chaque strate dans le même graphique.
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

    # 1. Chargement et Empilement
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

    # 2. Création de la Vérité Terrain (Y)
    # La première image sert de référence géométrique
    mask_classes = rasterize_samples(list_files[0], shp_path, field_name=field_name)

    # 3. Extraction des Pixels d'Entraînement
    # Garde que les pixels où le masque > 0 (classe existe)
    valid_pixels = (mask_classes > 0)
    # Extraction de X (Tableau des variables)
    X = X_full[valid_pixels, :]
    # Extraction de Y (Tableau des classes)
    Y = mask_classes[valid_pixels]
    
    # 4. Nettoyage final (NaN -> 0)
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

    # Partie 1 : L'Image
    print(f"\n1. RÉALITÉ TERRAIN (X_full) -> Pour la Carte Finale")
    print(f"   • Structure  : CUBE 3D (Lignes, Colonnes, Variables)")
    print(f"   • Dimensions : {X_full.shape}")
    print(f"   • Surface    : {total_pixels_image:,} pixels totaux".replace(',', ' '))
    print(f"   • Variables  : {X_full.shape[2]}")

    print("-" * 60)

    # Partie 2 : Le Dataset
    print(f"2. JEU D'ENTRAÎNEMENT (X, Y) -> Pour le Modèle")
    print(f"   • Structure  : TABLEAU 2D (Échantillons, Variables)")
    print(f"   • Dimensions X : {X.shape}")
    print(f"   • Dimensions Y : {Y.shape}")
    print(f"   • Échantillons : {total_pixels_train:,} pixels annotés".replace(',', ' '))

    print("-" * 60)

    # Partie 3 : Analyse
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

def plot_hyperparam_impact(cv_results):
    """
    Affiche l'impact des hyperparamètres (RF) sur le score moyen.
    Affichage uniquement (pas de sauvegarde).
    """
    df_results = pd.DataFrame(cv_results)
    
    # Configuration des paramètres à surveiller
    params_config = [
        ('param_n_estimators', "Nombre d'arbres"),
        ('param_max_depth', "Profondeur Max"),
        ('param_max_features', "Max Features"),
        ('param_min_samples_leaf', "Min Samples Leaf")
    ]
    
    # On ne garde que les paramètres présents dans les résultats
    active_params = [(p, t) for p, t in params_config if p in df_results.columns]
    
    if not active_params:
        print("Aucun paramètre standard trouvé dans les résultats.")
        return

    # Création dynamique de la figure
    nb_plots = len(active_params)
    fig, axes = plt.subplots(1, nb_plots, figsize=(5 * nb_plots, 5), sharey=True)
    
    # Gestion du cas où il n'y a qu'un seul graphique
    if nb_plots == 1: axes = [axes]

    for i, (param, title) in enumerate(active_params):
        ax = axes[i]
        
        # Gestion des valeurs (ex: None pour max_depth)
        temp_col = df_results[param].fillna('None')
        
        # Regroupement et Moyenne
        grouped = df_results.groupby(temp_col)['mean_test_score'].mean()
        
        # Tri de l'axe X si possible
        try: grouped = grouped.sort_index()
        except: pass

        # Préparation des axes
        x_vals = [str(x) for x in grouped.index]
        y_vals = grouped.values
        
        # --- TON STYLE ---
        ax.plot(x_vals, y_vals, marker='o', linewidth=2, linestyle='-', color='royalblue')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Annotations (Valeurs au-dessus des points)
        for x, y in zip(x_vals, y_vals):
            ax.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)

    # Titre global et Label Y sur le premier graph uniquement
    axes[0].set_ylabel("Précision Moyenne (Accuracy)")
    plt.suptitle("Impact moyen de chaque hyperparamètre sur la performance", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()

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

    # 1. Génération des noms
    for f in file_list:
        filename = os.path.basename(f)
        # Ouvre juste pour compter les bandes
        ds = rw.open_image(f)
        nb_bands = ds.RasterCount
        ds = None # Fermeture immédiate
        
        for i in range(nb_bands):
            if dates_list is not None and len(dates_list) == nb_bands:
                date_suffix = str(dates_list[i]).split(' ')[0] 
                expanded_feature_names.append(f"{filename}_{date_suffix}")
            else:
                # Sinon nom générique (ex: "Image_Date1")
                expanded_feature_names.append(f"{filename}_Date{i+1}")

    # 2. Vérifications de cohérence
    nb_names = len(expanded_feature_names)
    nb_imp = len(model.feature_importances_)
    
    print(f"Nombre de variables nommées    : {nb_names}")
    print(f"Nombre d'importances du modèle : {nb_imp}")
    
    if nb_names != nb_imp:
        print("ATTENTION : Le nombre de noms ne correspond pas au nombre de variables du modèle !")
    
    # 3. Création du DataFrame
    df_importance = pd.DataFrame({
        'Variable': expanded_feature_names,
        'Importance': model.feature_importances_
    })
    
    # 4. Tri décroissant
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
    
    # 1. Création du Masque de validité
    # Utilisation de l'image de référence sur le disque pour retrouver les vrais NaNs/NoData
    ds_ref = rw.open_image(ref_image_path)
    ref_band = ds_ref.GetRasterBand(1).ReadAsArray()
    file_nodata = ds_ref.GetRasterBand(1).GetNoDataValue()
    
    if file_nodata is not None:
        valid_mask = (ref_band != file_nodata)
    else:
        valid_mask = ~np.isnan(ref_band)
        
    print(f"Dimensions : {rows}x{cols}")
    
    # 2. Préparation des données
    # Aplatissement (Reshape) : Transforme le cube en tableau 2D
    X_flat = X_full.reshape(rows * cols, bands)
    mask_flat = valid_mask.reshape(rows * cols)
    
    # Ne garde QUE les pixels intéressants pour le modèle
    X_to_predict = X_flat[mask_flat]
    
    # Sécurité : Remplace les NaN par 0
    X_to_predict = np.nan_to_num(X_to_predict, nan=0.0)
    
    print(f"Pixels à prédire : {X_to_predict.shape[0]} (sur {rows*cols} totaux)")
    
    # 3. Prédiction (Uniquement sur les pixels valides)
    if X_to_predict.shape[0] > 0:
        Y_predicted = model.predict(X_to_predict)
    else:
        print("Attention : Aucun pixel valide trouvé !")
        return

    # 4. Reconstruction de l'image complète
    # Création image vide remplie de 0
    Y_full_flat = np.zeros(rows * cols, dtype=np.uint8)
    
    # Insertion des prédictions SEULEMENT aux bons endroits
    Y_full_flat[mask_flat] = Y_predicted
    
    # Redonne la forme 2D
    carte_strates = Y_full_flat.reshape(rows, cols)
    
    # 5. Sauvegarde
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
    # 1. Chargement de l'image
    ds = gdal.Open(map_path)
    data = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    
    # 2. Préparation des couleurs
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
    
    # 3. Affichage
    plt.figure(figsize=(15, 10))
    plt.imshow(data, cmap=cmap, interpolation='nearest')
    plt.title(title, fontsize=15, fontweight='bold')
    plt.axis('off') # Retire les axes X/Y
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()