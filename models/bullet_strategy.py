"""
Module pour la stratégie obligataire "Bullet".
"""

import pandas as pd
import numpy as np
from models.base_strategy import BondStrategy


class BulletStrategy(BondStrategy):
    """
    Implémentation de la stratégie obligataire "Bullet".
    
    La stratégie Bullet consiste à concentrer les investissements sur une maturité cible spécifique,
    plutôt que de les répartir sur plusieurs échéances comme dans les stratégies Laddering ou Barbell.
    Cette approche permet de minimiser le risque de réinvestissement et de cibler précisément
    un point sur la courbe des taux.
    """
    
    def __init__(self, target_maturity=5, maturity_range=1, **kwargs):
        """
        Initialise une stratégie Bullet.
        
        Args:
            target_maturity (int, optional): Maturité cible en années. Par défaut 5.
            maturity_range (int, optional): Plage de maturité autour de la cible (en années). Par défaut 1.
            **kwargs: Arguments supplémentaires à passer à la classe parent.
        """
        super().__init__(name="Bullet Strategy", **kwargs)
        
        self.target_maturity = target_maturity
        self.maturity_range = maturity_range
        
        # Mapping des maturités aux ETF correspondants
        self.maturity_etf_map = {
            1: "SHY",    # iShares 1-3 Year Treasury Bond ETF
            2: "SHY",    # iShares 1-3 Year Treasury Bond ETF
            3: "IEI",    # iShares 3-7 Year Treasury Bond ETF
            5: "IEI",    # iShares 3-7 Year Treasury Bond ETF
            7: "IEF",    # iShares 7-10 Year Treasury Bond ETF
            10: "IEF",   # iShares 7-10 Year Treasury Bond ETF
            15: "GOVT",  # iShares U.S. Treasury Bond ETF (mix de maturités)
            20: "TLT",   # iShares 20+ Year Treasury Bond ETF
            30: "TLT"    # iShares 20+ Year Treasury Bond ETF
        }
        
    def get_etfs_in_range(self):
        """
        Détermine les ETF à utiliser pour la maturité cible et sa plage.
        
        Returns:
            list: Liste des ETF à utiliser.
        """
        min_maturity = self.target_maturity - self.maturity_range
        max_maturity = self.target_maturity + self.maturity_range
        
        # S'assurer que les maturités sont positives
        min_maturity = max(1, min_maturity)
        
        # Trouver les ETF pour chaque maturité dans la plage
        etfs = set()
        for maturity in range(min_maturity, max_maturity + 1):
            closest_maturity = min(self.maturity_etf_map.keys(), key=lambda x: abs(x - maturity))
            etfs.add(self.maturity_etf_map[closest_maturity])
            
        return list(etfs)
        
    def generate_weights(self, data, date):
        """
        Génère les poids pour chaque ETF obligataire à une date donnée,
        en fonction de la stratégie bullet.
        
        Args:
            data (pandas.DataFrame): Données des ETF obligataires.
            date (datetime): Date pour laquelle générer les poids.
            
        Returns:
            pandas.Series: Poids à allouer à chaque ETF.
        """
        # Si la date n'est pas dans les données, prendre la date la plus proche
        if date not in data.index:
            closest_date = data.index[data.index.get_indexer([date], method='nearest')[0]]
            date = closest_date
            
        # Obtenir les ETF pour la maturité cible
        target_etfs = self.get_etfs_in_range()
        
        # Vérifier quels ETF sont disponibles
        available_etfs = [etf for etf in target_etfs if etf in data.columns]
        
        if not available_etfs:
            raise ValueError(f"Aucun ETF disponible pour la stratégie Bullet avec maturité cible {self.target_maturity}")
        
        # Créer le dictionnaire de poids avec répartition égale
        weights = {etf: 1.0 / len(available_etfs) for etf in available_etfs}
        
        return pd.Series(weights)
    
    def adjust_target_for_yield_curve(self, yield_curve):
        """
        Ajuste la maturité cible en fonction de la forme de la courbe des taux
        pour maximiser le rendement par unité de risque.
        
        Args:
            yield_curve (pandas.Series): Courbe des taux avec les rendements par maturité.
            
        Returns:
            int: Nouvelle maturité cible ajustée.
        """
        # Extraire les rendements et maturités de la courbe des taux
        maturities = []
        yields = []
        
        # Convertir les étiquettes de maturité en nombres
        maturity_map = {'3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}
        
        for label, value in yield_curve.items():
            if label in maturity_map:
                maturities.append(maturity_map[label])
                yields.append(value)
                
        if not maturities:
            return self.target_maturity  # Pas de changement si pas de données
            
        # Calculer la pente entre chaque point
        slopes = []
        for i in range(1, len(maturities)):
            slope = (yields[i] - yields[i-1]) / (maturities[i] - maturities[i-1])
            slopes.append((maturities[i], slope))
            
        if not slopes:
            return self.target_maturity  # Pas de changement si pas assez de données
            
        # Identifier le point où la pente commence à diminuer significativement
        # Ce point offre généralement le meilleur compromis rendement/risque
        max_slope = max(slopes, key=lambda x: x[1])
        optimal_maturity_index = slopes.index(max_slope)
        
        # Prendre la maturité juste après le point de pente maximale
        if optimal_maturity_index + 1 < len(maturities):
            new_target = maturities[optimal_maturity_index + 1]
        else:
            new_target = maturities[-1]  # Dernière maturité si on est à la fin
            
        # Arrondir à l'entier le plus proche et limiter à des valeurs raisonnables
        new_target = min(30, max(1, round(new_target)))
        
        return new_target
    
    def adjust_for_economic_cycle(self, cycle_indicator):
        """
        Ajuste la maturité cible en fonction du cycle économique.
        
        Args:
            cycle_indicator (str): Indicateur du cycle économique ('expansion', 'peak', 'contraction', 'trough').
            
        Returns:
            int: Nouvelle maturité cible ajustée.
        """
        if cycle_indicator == 'expansion':
            # En expansion, privilégier les maturités courtes à moyennes car les taux pourraient monter
            return min(5, self.target_maturity)
        elif cycle_indicator == 'peak':
            # Au sommet du cycle, privilégier les maturités courtes car les taux sont généralement élevés
            return min(3, self.target_maturity)
        elif cycle_indicator == 'contraction':
            # En contraction, privilégier les maturités moyennes à longues car les taux pourraient baisser
            return max(7, self.target_maturity)
        elif cycle_indicator == 'trough':
            # Au creux du cycle, privilégier les maturités longues pour verrouiller les rendements
            return max(10, self.target_maturity)
        else:
            # Pas de changement si l'indicateur n'est pas reconnu
            return self.target_maturity
