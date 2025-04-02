"""
Module de chargement des données obligataires et économiques pour les stratégies de portefeuille.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta

class DataLoader:
    """
    Classe pour charger et prétraiter les données obligataires et économiques.
    """
    
    def __init__(self, api_key=None, cache_dir="data/cache"):
        """
        Initialise le DataLoader.
        
        Args:
            api_key (str, optional): Clé API pour FRED. Par défaut à None.
            cache_dir (str, optional): Répertoire pour le cache des données. Par défaut à "data/cache".
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        
        # S'assurer que le répertoire de cache existe
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        # Initialiser l'API FRED si une clé est fournie
        if api_key:
            self.fred = Fred(api_key=api_key)
        else:
            self.fred = None
            
    def load_treasury_yields(self, start_date=None, end_date=None):
        """
        Charge les rendements des bons du Trésor américain.
        
        Args:
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à 10 ans avant aujourd'hui.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à aujourd'hui.
            
        Returns:
            pandas.DataFrame: DataFrame contenant les rendements des bons du Trésor.
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # Définir les tickers pour les rendements des bons du Trésor
        tickers = [
            "^IRX",  # 13-Week Treasury Bill
            "^FVX",  # 5-Year Treasury Note
            "^TNX",  # 10-Year Treasury Note
            "^TYX"   # 30-Year Treasury Bond
        ]
        
        cache_file = f"{self.cache_dir}/treasury_yields_{start_date}_{end_date}.csv"
        
        # Vérifier si nous avons des données en cache
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Télécharger les données
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Renommer les colonnes
        data.columns = ['3M', '5Y', '10Y', '30Y']
        
        # Sauvegarder dans le cache
        data.to_csv(cache_file)
        
        return data
    
    def load_fed_rates(self, start_date=None, end_date=None):
        """
        Charge les taux de la Fed à partir de l'API FRED.
        
        Args:
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à 10 ans avant aujourd'hui.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à aujourd'hui.
            
        Returns:
            pandas.DataFrame: DataFrame contenant les taux de la Fed.
        """
        if not self.fred:
            raise ValueError("FRED API key is required to load Fed rates")
            
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        cache_file = f"{self.cache_dir}/fed_rates_{start_date}_{end_date}.csv"
        
        # Vérifier si nous avons des données en cache
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Télécharger les données
        fed_funds_rate = self.fred.get_series('FEDFUNDS', observation_start=start_date, observation_end=end_date)
        
        # Convertir en DataFrame
        data = pd.DataFrame(fed_funds_rate, columns=['FedFundsRate'])
        
        # Sauvegarder dans le cache
        data.to_csv(cache_file)
        
        return data
    
    def load_bond_etfs(self, etf_list=None, start_date=None, end_date=None):
        """
        Charge les données de prix et de volume pour les ETF obligataires.
        
        Args:
            etf_list (list, optional): Liste des symbols des ETF. Par défaut à une liste prédéfinie.
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à 10 ans avant aujourd'hui.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à aujourd'hui.
            
        Returns:
            pandas.DataFrame: DataFrame contenant les prix ajustés des ETF.
        """
        if not etf_list:
            # ETF obligataires par défaut à suivre
            etf_list = [
                "AGG",   # iShares Core U.S. Aggregate Bond ETF
                "BND",   # Vanguard Total Bond Market ETF
                "GOVT",  # iShares U.S. Treasury Bond ETF
                "SHY",   # iShares 1-3 Year Treasury Bond ETF
                "IEF",   # iShares 7-10 Year Treasury Bond ETF
                "TLT",   # iShares 20+ Year Treasury Bond ETF
                "LQD",   # iShares iBoxx $ Investment Grade Corporate Bond ETF
                "HYG",   # iShares iBoxx $ High Yield Corporate Bond ETF
                "MUB",   # iShares National Muni Bond ETF
                "VCSH"   # Vanguard Short-Term Corporate Bond ETF
            ]
            
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        etf_str = "_".join(etf_list)
        cache_file = f"{self.cache_dir}/bond_etfs_{etf_str}_{start_date}_{end_date}.csv"
        
        # Vérifier si nous avons des données en cache
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Télécharger les données
        data = yf.download(etf_list, start=start_date, end=end_date)['Adj Close']
        
        # Sauvegarder dans le cache
        data.to_csv(cache_file)
        
        return data
    
    def load_economic_indicators(self, indicators=None, start_date=None, end_date=None):
        """
        Charge les indicateurs économiques à partir de l'API FRED.
        
        Args:
            indicators (dict, optional): Dictionnaire des indicateurs à charger {nom: code FRED}.
                                        Par défaut à un ensemble prédéfini d'indicateurs.
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à 10 ans avant aujourd'hui.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à aujourd'hui.
            
        Returns:
            pandas.DataFrame: DataFrame contenant les indicateurs économiques.
        """
        if not self.fred:
            raise ValueError("FRED API key is required to load economic indicators")
            
        if not indicators:
            # Indicateurs économiques par défaut
            indicators = {
                'GDP': 'GDP',                         # Gross Domestic Product
                'CPI': 'CPIAUCSL',                    # Consumer Price Index
                'Unemployment': 'UNRATE',             # Unemployment Rate
                'Industrial_Production': 'INDPRO',    # Industrial Production Index
                'Retail_Sales': 'RSXFS',              # Retail Sales
                'Housing_Starts': 'HOUST',            # Housing Starts
                'Consumer_Sentiment': 'UMCSENT'       # Consumer Sentiment Index
            }
            
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        indicators_str = "_".join(indicators.keys())
        cache_file = f"{self.cache_dir}/economic_indicators_{indicators_str}_{start_date}_{end_date}.csv"
        
        # Vérifier si nous avons des données en cache
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Télécharger les données
        data = pd.DataFrame()
        
        for name, code in indicators.items():
            series = self.fred.get_series(code, observation_start=start_date, observation_end=end_date)
            data[name] = series
            
        # Certains indicateurs peuvent avoir des fréquences différentes, nous résamplons à la fréquence mensuelle
        data = data.resample('M').last()
        
        # Remplir les valeurs manquantes avec la méthode forward-fill
        data = data.fillna(method='ffill')
        
        # Sauvegarder dans le cache
        data.to_csv(cache_file)
        
        return data

    def load_benchmark_data(self, ticker="AGG", start_date=None, end_date=None):
        """
        Charge les données d'un indice de référence.
        
        Args:
            ticker (str, optional): Symbol de l'indice. Par défaut à "AGG" (iShares Core U.S. Aggregate Bond ETF).
            start_date (str, optional): Date de début au format 'YYYY-MM-DD'. Par défaut à 10 ans avant aujourd'hui.
            end_date (str, optional): Date de fin au format 'YYYY-MM-DD'. Par défaut à aujourd'hui.
            
        Returns:
            pandas.DataFrame: DataFrame contenant les données de l'indice.
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        cache_file = f"{self.cache_dir}/benchmark_{ticker}_{start_date}_{end_date}.csv"
        
        # Vérifier si nous avons des données en cache
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Télécharger les données
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Sauvegarder dans le cache
        data.to_csv(cache_file)
        
        return data

    def calculate_yield_curve(self, date=None):
        """
        Calcule la courbe des taux pour une date donnée.
        
        Args:
            date (str, optional): Date au format 'YYYY-MM-DD'. Par défaut à la dernière date disponible.
            
        Returns:
            pandas.Series: Série contenant la courbe des taux.
        """
        # Charger les rendements des bons du Trésor
        yields_data = self.load_treasury_yields()
        
        if date:
            # Sélectionner la date la plus proche
            date = pd.to_datetime(date)
            closest_date = yields_data.index[yields_data.index.get_indexer([date], method='nearest')[0]]
            yield_curve = yields_data.loc[closest_date]
        else:
            # Prendre la dernière date disponible
            yield_curve = yields_data.iloc[-1]
            
        return yield_curve
